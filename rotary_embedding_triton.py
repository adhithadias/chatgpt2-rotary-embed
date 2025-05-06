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
    BLOCK_D: tl.constexpr, # optimize the kernel assuming it is fixed.
    conj: tl.constexpr # conjugate flag
):
    
    # Triton allows launching a grid of programs in multiple dimensions.
    b = tl.program_id(0)    # Batch index
    s = tl.program_id(1)    # Sequence index
    h = tl.program_id(2)    # Head index

    # [0, 1, 2, 3]
    offs_d = tl.arange(0, BLOCK_D)  # Vector of offsets for the feature dimension.

    x_offset = b * stride_bs + s * stride_sh + h * stride_hd   # flattened index

    # Cos/sin are [S, D//2]
    cos_offset = s * (D // 2) + offs_d
    mask_half = offs_d < (D // 2)  # Handle tail fragments when D is not a multiple of BLOCK_D
    cos = tl.load(cos_ptr + cos_offset, mask=mask_half, other=0.0)
    sin = tl.load(sin_ptr + cos_offset, mask=mask_half, other=0.0)

    offs_even = 2 * offs_d  # [0, 2, 4, 6]
    x_even = tl.load(x_ptr + offs_even, mask=offs_even < D)     # Even Values

    offs_odd = 2 * offs_d + 1   # [1, 3, 5, 7]
    x_odd = tl.load(x_ptr + offs_odd, mask=offs_odd < D)       # Odd Values

    # Triton does not support complex numbers natively, so we'll have to use the sin-cos approach
    if not conj:        # Forward pass
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
    else:               # Backward pass
        x_rotated_even = x_even * cos + x_odd * sin
        x_rotated_odd = -x_even * sin + x_odd * cos

    # Interleave
    offs = tl.arange(0, BLOCK_D)
    offs_even = offs_d * 2
    offs_odd  = offs_d * 2 + 1
    # x_offset = x_offset + tl.zeros([BLOCK_D // 2], dtype=tl.int32)  # shape match!

    # Compute final store addresses
    store_ptr_even = out_ptr + x_offset + offs_even
    store_ptr_odd  = out_ptr + x_offset + offs_odd

    # Store interleaved output
    mask_even = offs_even < D
    mask_odd = offs_odd < D
    tl.store(store_ptr_even, x_rotated_even, mask=mask_even)
    tl.store(store_ptr_odd,  x_rotated_odd,  mask=mask_odd)



# def rotary_kernel(
#     x_ptr, cos_ptr, sin_ptr, out_ptr, conj,
#     B, S, H, D,
#     stride_bs, stride_sh, stride_hd,
#     BLOCK_D: tl.constexpr
# ):
#     b = tl.program_id(0)
#     s = tl.program_id(1)
#     h = tl.program_id(2)

#     x_offset = b * stride_bs + s * stride_sh + h * stride_hd

#     offs_d = tl.arange(0, BLOCK_D)
#     offs_even = 2 * offs_d
#     offs_odd  = 2 * offs_d + 1

#     # Mask for even/odd loads/stores
#     mask_even = offs_even < D
#     mask_odd = offs_odd < D

#     # Load x1 (even) and x2 (odd)
#     x1 = tl.load(x_ptr + x_offset + offs_even, mask=mask_even, other=0.0)
#     x2 = tl.load(x_ptr + x_offset + offs_odd, mask=mask_odd, other=0.0)

#     # Load cos/sin for this sequence position
#     mask_rot = offs_d < (D // 2)
#     cos = tl.load(cos_ptr + s * (D // 2) + offs_d, mask=mask_rot, other=0.0)
#     sin = tl.load(sin_ptr + s * (D // 2) + offs_d, mask=mask_rot, other=0.0)

#     # Apply rotation
#     if not conj:
#         x_rotated_even = x1 * cos - x2 * sin
#         x_rotated_odd  = x1 * sin + x2 * cos
#     else:
#         x_rotated_even = x1 * cos + x2 * sin
#         x_rotated_odd  = -x1 * sin + x2 * cos

#     # Store back with masks
#     tl.store(out_ptr + x_offset + offs_even, x_rotated_even, mask=mask_even)
#     tl.store(out_ptr + x_offset + offs_odd,  x_rotated_odd,  mask=mask_odd)


def apply_rotary(x, cos, sin, out=None, conj=False):
    B, S, H, D = x.shape
    if out is None:
        out = torch.empty_like(x)
        # out = out.contiguous()
    
    # assert x.is_contiguous()
    # assert cos.is_contiguous() and sin.is_contiguous()
    # assert out.is_contiguous()
    # assert x.device == cos.device == sin.device
    # assert x.dtype == cos.dtype == sin.dtype
    grid = (B, S, H)
    rotary_kernel[grid](
        x, cos, sin, out,
        B, S, H, D,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_D=triton.next_power_of_2(D),       # Set the feature dimension to a power of 2
        conj=conj
    )
    return out
