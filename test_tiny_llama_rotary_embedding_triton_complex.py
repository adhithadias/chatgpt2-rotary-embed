from typing import Tuple
import torch
from dataclasses import dataclass
from rotary_embedding import apply_rotary_emb_triton2
import time
import nvtx

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--block_size', type=int, required=True)
args = parser.parse_args()

batch_size = args.batch_size
# block_size = args.block_size


BENCHMARK_FREQUENCY = 100
# batch_size = 32

# dtype = torch.float32
dtype = torch.bfloat16

model_type = 'gpt2'  # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

config_args = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
}[model_type]

@dataclass
class GPTConfig:
    block_size: int = args.block_size # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    rotary_percentage: float = 1.0 # percentage of rotary embedding

RoPECache = Tuple[torch.Tensor]
def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    # print(n_elem)
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    # print('theta.shape:', theta.shape)
    # print('theta:', theta)

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    
    # print('cos.dtype:', cos.dtype)
    print('idx_theta.dtype:', idx_theta.dtype)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    # if dtype == torch.bfloat16:
    #     return cos.bfloat16(), sin.bfloat16()
    if dtype == torch.bfloat16:
        return idx_theta.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    # if dtype in (torch.float16, torch.bfloat16, torch.int8):
    #     return cos.half(), sin.half()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return idx_theta.half()
    
    # print('cos.dtype:', cos.dtype)
    return idx_theta


# def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
#     return build_rope_cache(
#         seq_len=self.config.block_size,
#         n_elem=int(self.config.rotary_percentage * self.config.head_size),
#         dtype=torch.bfloat16,
#         device=idx.device,
#         condense_ratio=self.config.condense_ratio,
#     )

if torch.cuda.is_available():
    device = torch.device('cuda')

config = GPTConfig(**config_args)

theta = build_rope_cache(
    seq_len=config.block_size,
    n_elem=int(config.rotary_percentage * config.n_embd // config.n_head),
    dtype=dtype,
    device=device,
    condense_ratio=1,
)

# theta = build_rope_cache(
#     seq_len=config.block_size,
#     n_elem=int(config.rotary_percentage * config.n_embd // config.n_head),
#     dtype=dtype,
#     device=device,
#     condense_ratio=1,
# )

# print('theta shape', theta.shape)  # Expected output: torch.Size([1024, 64])
# print(sin[1,:])
# print(cos.shape)  # Expected output: torch.Size([1024, 64])
# print(cos[1,:])

if torch.cuda.is_available():
    device = torch.device('cuda')

# (batch_size, seq_len, n_head, head_dim)
# head_dim = n_embd // n_head, 768 // 12 = 64
# q = torch.randn(2, 1024, 12, 64).to(device=device, dtype=torch.float32)
# k = torch.randn(2, 1024, 12, 64).to(device=device, dtype=torch.float32)

# # create tensor with all ones
# q = torch.ones(2, 1024, 12, 64).to(device=device, dtype=torch.float32)
# k = torch.ones(2, 1024, 12, 64).to(device=device, dtype=torch.float32)

# print('cos\n', cos)
# print('sin\n', sin)

base_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
total_elements = batch_size * config.block_size * config.n_head * (config.n_embd // config.n_head)

q = base_tensor.repeat(total_elements // base_tensor.numel()).reshape(batch_size, config.block_size, config.n_head, (config.n_embd // config.n_head)).to(dtype=dtype, device=device)
k = base_tensor.repeat(total_elements // base_tensor.numel()).reshape(batch_size, config.block_size, config.n_head, (config.n_embd // config.n_head)).to(dtype=dtype, device=device)

# print('q:\n', q)

# print('q.shape:', q.shape)
# print('q.dtype:', q.dtype)

execution_times = []

# apply rope in fp32 significanly stabalize training
# fused rope expect (batch_size, seqlen, nheads, headdim)
torch.cuda.synchronize()
for i in range(BENCHMARK_FREQUENCY):
    # add nvtx annotation
    t1 = time.time()
    # execute without gradient tracking
    with torch.no_grad():
        # start = nvtx.start_range(message="custom_rope", color="blue")
        
        # xq = apply_rotary_emb_func3(q, cos, sin, True, False)
        # xk = apply_rotary_emb_func3(k, cos, sin, True, False)

        xq = apply_rotary_emb_triton2(q, theta, True, False)
        xk = apply_rotary_emb_triton2(k, theta, True, False)
        
        torch.cuda.synchronize()
        # nvtx.end_range(start)
    # include cuda synchronize
    t2 = time.time()
    
    # print time in nano seconds
    time_ms = (t2 - t1) * 1e6
    execution_times.append(time_ms)
    # print('time taken:', (t2 - t1) * 1e6, 'us')
    # print(f"Time taken: {t2 - t1:.4f} seconds")

# print('=========')
# print(q)
# print('=========')
# print(xq)


# find median of execution time
median_time = sorted(execution_times)[len(execution_times) // 2]

# print median time with 2 decimal places
print('median time taken:', f"{median_time:.2f}", 'us')