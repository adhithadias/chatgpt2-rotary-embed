import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    x_ptr,          # [B, S, H, D] — interleaved real-imag
    theta_ptr,      # [D/2] — rotation frequencies
    out_ptr,        # output tensor
    B, S, H,
    D: tl.constexpr,
    stride_bs, stride_sh, stride_hd,
    BLOCK_D: tl.constexpr,
    conj: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)


    offs_d = tl.arange(0, BLOCK_D)  # Vector of offsets for the feature dimension.

    x_offset = b * stride_bs + s * stride_sh + h * stride_hd   # flattened index

    # Theta is [S, D//2]
    
    theta_offset = s * (D // 2) + offs_d
    mask_half = offs_d < (D // 2)  # Handle tail fragments when D is not a multiple of BLOCK_D
    theta = tl.load(theta_ptr + theta_offset, mask=mask_half, other=0.0)

    angle = theta * s
    cos_val = tl.cos(angle)
    sin_val = tl.sin(angle)

    offs_even = 2 * offs_d

    x_even = tl.load(x_ptr + offs_even, mask=offs_even < D)     # Even Values
    x_odd = tl.load(x_ptr + offs_even + 1, mask=offs_even + 1 < D)       # Odd Values

    # Forward pass and backward pass using triton's where and value of conj
    out_even = tl.where(conj, x_even * cos_val + x_odd * sin_val, x_even * cos_val - x_odd * sin_val)
    out_odd  = tl.where(conj, -x_even * sin_val + x_odd * cos_val, x_even * sin_val + x_odd * cos_val)

    offs_even = offs_d * 2

    store_ptr_even = out_ptr + x_offset + offs_even

    tl.store(store_ptr_even, out_even, mask=offs_even < D)
    tl.store(store_ptr_even + 1,  out_odd,  mask=offs_even + 1 < D)


def apply_rotary_direct_theta(x, theta, out=None, conj=False):
    B, S, H, D = x.shape
    if out is None:
        out = torch.empty_like(x)

    grid = (B, S, H)
    rotary_kernel[grid](
        x, theta, out,
        B, S, H, D,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_D=triton.next_power_of_2(D),
        conj=conj,
        num_warps=2, num_stages=2
    )
    return out
