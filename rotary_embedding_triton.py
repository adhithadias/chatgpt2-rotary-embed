import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    x_ptr,           # pointer to x
    cos_ptr,         # pointer to cos
    sin_ptr,         # pointer to sin
    out_ptr,         # pointer to output tensor
    B, S, H, D,      # batch, sequence, heads, dimension                #2, 2, 2, 4
    stride_bs, stride_sh, stride_hd,  # strides                         #16, 8, 4
    BLOCK_D: tl.constexpr # optimize the kernel assuming it is fixed.
):
    
    # Triton allows launching a grid of programs in multiple dimensions.
    b = tl.program_id(0)    # Batch index
    s = tl.program_id(1)    # Sequence index
    h = tl.program_id(2)    # Head index

    # [0, 1, 2, 3]
    offs_d = tl.arange(0, BLOCK_D)  # Vector of offsets for the feature dimension.
    mask = offs_d < D               # Handle tail fragments when D is not a multiple of BLOCK_D

    
    x_offset = b * stride_bs + s * stride_sh + h * stride_hd + offs_d   # flattened index
    # x = tl.load(x_ptr + x_offset, mask=mask)

    # Cos/sin are [S, D//2]
    cos = tl.load(cos_ptr + s * (BLOCK_D//2) + offs_d, mask=mask)
    sin = tl.load(sin_ptr + s * (BLOCK_D//2) + offs_d, mask=mask)

    # x1 = x[::2]     # Even Values
    # x2 = x[1::2]    # Odd Values
    offs_even = 2 * offs_d  # [0, 2, 4, 6]
    x1 = tl.load(x_ptr + offs_even, mask=offs_even < D)     # Even Values

    offs_odd = 2 * offs_d + 1   # [1, 3, 5, 7]
    x2 = tl.load(x_ptr + offs_odd, mask=offs_odd < D)       # Odd Values

    # offs = tl.arange(0, BLOCK_D // 2)  # NOT BLOCK_D
    # offsets_even = offs * 2
    # offsets_odd = offsets_even + 1

    # x1 = tl.load(x_ptr + offsets_even, ...)   # shape: [BLOCK_D // 2]
    # x2 = tl.load(x_ptr + offsets_odd, ...)    # shape: [BLOCK_D // 2]

    # cos = tl.load(cos_ptr + s * (D // 2) + offs, ...)  # shape: [BLOCK_D // 2]
    # sin = tl.load(sin_ptr + s * (D // 2) + offs, ...)  # shape: [BLOCK_D // 2]

    # Triton does not support complex numbers natively, so we'll have to use the real-img approach
    x_rotated_even = x1 * cos - x2 * sin
    x_rotated_odd = x1 * sin + x2 * cos

    # # Interleave
    # out = tl.zeros([BLOCK_D], dtype=tl.float32)
    # out[::2] = x_rotated_even
    # out[1::2] = x_rotated_odd

    # Interleaving positions
    offs = tl.arange(0, BLOCK_D)
    offs_even = offs * 2
    offs_odd  = offs * 2 + 1
    x_offset = b * stride_bs + s * stride_sh + h * stride_hd
    # x_offset = x_offset + tl.zeros([BLOCK_D // 2], dtype=tl.int32)  # shape match!

    # Compute final store addresses
    store_ptr_even = out_ptr + x_offset + offs_even
    store_ptr_odd  = out_ptr + x_offset + offs_odd

    # tl.store(out_ptr + x_offset + offs_even, x_rotated_even)
    # tl.store(out_ptr + x_offset + offs_odd,  x_rotated_odd)

    # Store interleaved output
    tl.store(store_ptr_even, x_rotated_even)
    tl.store(store_ptr_odd,  x_rotated_odd)


    # tl.store(out_ptr + x_offset, out, mask=mask)


def apply_rotary(x, cos, sin, out=None):
    B, S, H, D = x.shape
    if out is None:
        out = torch.empty_like(x)
    
    grid = (B, S, H)
    rotary_kernel[grid](
        x, cos, sin, out,
        B, S, H, D,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_D=triton.next_power_of_2(D)       # Set the feature dimension to a power of 2
    )
    return out