'''
Implementation of rotary embedding taken from llama
'''

from dataclasses import dataclass
from typing import Tuple
import torch
import pdb
# import time
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
# batch_size = 8
dtype = torch.bfloat16
if torch.cuda.is_available():
    device = torch.device('cuda')

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
    
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    # print('ndim', ndim)
    # print(x.shape)
    # print(freqs_cis.shape)
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)    

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # print('xq', xq.shape)
    # print('xk', xk.shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # print(xq_)
    # print('xq_', xq_.shape)
    # print('xk_.shape:', xk_.shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # print('freqs_cis.shape:', freqs_cis.shape)
    # print('//////////')
    # print(freqs_cis)
    # print('freqs_cis device', freqs_cis.device)
    # print('xq_ device', xq_.device)
    # xq_out = xq_ * freqs_cis
    # print('==========')
    # print(xq_out)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(
    dim: int, end: int, 
    theta: float = 10000.0, 
    device=None,
    dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    # print(dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # print(freqs.shape)
    # print('freqs:', freqs)
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # print('seq_idx.shape:', t.shape)
    # print(t)
    freqs = torch.outer(t, freqs) # type: ignore
    # convert datatype to bfloat16
    freqs = freqs.to(dtype=torch.float32)
    # print('freqs.shape:', freqs.shape)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis = freqs_cis.to(device=device,dtype=dtype)
    return freqs_cis

config = GPTConfig(**config_args)

freqs_cis = precompute_freqs_cis(
    config.n_embd // config.n_head, 
    block_size, 
    device=device,
    dtype=dtype,
)

# print(freqs_cis.shape)  # Expected output: torch.Size([1024, 64])
# print(freqs_cis[1,:])

# (batch_size, seq_len, n_head, head_dim)
# head_dim = n_embd // n_head, 768 // 12 = 64
# xq = torch.randn(2, 1024, 12, 64)
# xk = torch.randn(2, 1024, 12, 64)

base_tensor = torch.tensor([1, 2, 3, 4])
total_elements = batch_size * block_size * config.n_head * (config.n_embd // config.n_head)

xq = base_tensor.repeat(total_elements // base_tensor.numel()).reshape(batch_size, block_size, config.n_head, config.n_embd // config.n_head).to(dtype=dtype, device=device)
xk = base_tensor.repeat(total_elements // base_tensor.numel()).reshape(batch_size, block_size, config.n_head, config.n_embd // config.n_head).to(dtype=dtype, device=device)

# print(xq)
# print(xq)
# exit(0)

# create tensor with all ones
# xq = torch.ones(2, 1024, 12, 64).to(dtype=dtype)
# xk = torch.ones(2, 1024, 12, 64).to(dtype=dtype)

# pdb.set_trace()

# xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

execution_times = []

for i in range(BENCHMARK_FREQUENCY):
    t1 = time.time()
    # execute without gradient tracking
    with torch.no_grad():
        # start = nvtx.start_range(message="matmul", color="blue")
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
        torch.cuda.synchronize()
        # nvtx.end_range(start)
    t2 = time.time()
    # print time in nano seconds
    time_ns = (t2 - t1) * 1e6
    execution_times.append(time_ns)
    # print('time taken:', (t2 - t1) * 1e6, 'ms')
    
# print(xq_out) 
# print(xq_out) 
    
# find median of execution time
median_time = sorted(execution_times)[len(execution_times) // 2]
print('median time taken:', f"{median_time:.2f}", 'us')

# pdb.set_trace()

# print('=========')