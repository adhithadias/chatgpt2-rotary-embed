from typing import Tuple
import torch
from dataclasses import dataclass
from rotary_embedding import apply_rotary_emb_func

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    rotary_percentage: float = 1.0 # percentage of rotary embedding

RoPECache = Tuple[torch.Tensor, torch.Tensor]

def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    print(n_elem)
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    print('theta.shape:', theta.shape)
    print('theta:', theta)

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    print('seq_idx.shape:', seq_idx.shape)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)
    print('idx_theta.shape:', idx_theta.shape)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    
    print('cos.dtype:', cos.dtype)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    
    print('cos.dtype:', cos.dtype)
    return cos, sin


# def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
#     return build_rope_cache(
#         seq_len=self.config.block_size,
#         n_elem=int(self.config.rotary_percentage * self.config.head_size),
#         dtype=torch.bfloat16,
#         device=idx.device,
#         condense_ratio=self.config.condense_ratio,
#     )
    
model_type = 'gpt2'  # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

config_args = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
}[model_type]

if torch.cuda.is_available():
    device = torch.device('cuda')

config = GPTConfig(**config_args)

sin, cos = build_rope_cache(
    seq_len=config.block_size,
    n_elem=int(config.rotary_percentage * config.n_embd // config.n_head),
    dtype=torch.float32,
    device=device,
    condense_ratio=1,
)

print(sin.shape)  # Expected output: torch.Size([1024, 64])
print(sin[1,:])
print(cos.shape)  # Expected output: torch.Size([1024, 64])
print(cos[1,:])

if torch.cuda.is_available():
    device = torch.device('cuda')

# (batch_size, seq_len, n_head, head_dim)
# head_dim = n_embd // n_head, 768 // 12 = 64
# q = torch.randn(2, 1024, 12, 64).to(device=device, dtype=torch.float32)
# k = torch.randn(2, 1024, 12, 64).to(device=device, dtype=torch.float32)

# # create tensor with all ones
# q = torch.ones(2, 1024, 12, 64).to(device=device, dtype=torch.float32)
# k = torch.ones(2, 1024, 12, 64).to(device=device, dtype=torch.float32)

base_tensor = torch.tensor([1, 2, 3, 4])
total_elements = 2 * 1024 * 12 * 64

q = base_tensor.repeat(total_elements // base_tensor.numel()).reshape(2, 1024, 12, 64).to(dtype=torch.float32, device=device)
k = base_tensor.repeat(total_elements // base_tensor.numel()).reshape(2, 1024, 12, 64).to(dtype=torch.float32, device=device)

print('q.dtype:', q.dtype)

# apply rope in fp32 significanly stabalize training
# fused rope expect (batch_size, seqlen, nheads, headdim)
q = apply_rotary_emb_func(q, cos, sin, True, True)
k = apply_rotary_emb_func(k, cos, sin, True, True)

print('=========')
print(q)