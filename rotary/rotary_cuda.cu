/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <torch/python.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
 

 void apply_rotary_cuda(const torch::Tensor x1, const torch::Tensor x2,
                        const torch::Tensor cos, const torch::Tensor sin,
                        torch::Tensor out1, torch::Tensor out2,
                        const bool conj) {
     auto iter = at::TensorIteratorConfig()
         .add_output(out1)
         .add_output(out2)
         .add_input(x1)
         .add_input(x2)
         .add_input(cos)
         .add_input(sin)
         .check_all_same_dtype(false)
         .promote_inputs_to_common_dtype(false)
         .build();

    int64_t numel = iter.numel();
    int64_t grid = (numel + block_work_size() - 1) / block_work_size();
    // std::cout << "numel: " << numel << std::endl;
    // std::cout << "Grid size: " << grid << std::endl;
    // std::cout << "num_threads: " << num_threads() << std::endl;
    // std::cout << "block work size: " << block_work_size() << std::endl;
 
     if (!conj) {
         AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x1.scalar_type(), "rotary_kernel", [&] {
             at::native::gpu_kernel_multiple_outputs(
                 iter, [] GPU_LAMBDA (scalar_t x1, scalar_t x2, scalar_t cos,
                                     scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                // interchanged sin and cos from the original implementation to match complex number implementation
                 scalar_t out1 = float(x1) * float(cos) - float(x2) * float(sin);
                 scalar_t out2 = float(x1) * float(sin) + float(x2) * float(cos);
                 return {out1, out2};
             });
         });
     } else {
         AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x1.scalar_type(), "rotary_kernel", [&] {
             at::native::gpu_kernel_multiple_outputs(
                 iter, [] GPU_LAMBDA (scalar_t x1, scalar_t x2, scalar_t cos,
                                     scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                 scalar_t out1 = float(x1) * float(cos) + float(x2) * float(sin);
                 scalar_t out2 = -float(x1) * float(sin) + float(x2) * float(cos);
                 return {out1, out2};
             });
         });
     }
 }

 __global__ void rotary_kernel_bfloat16(const __nv_bfloat16* x,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* out,
    const int64_t N, const bool conj, const int32_t S, const int32_t H, const int32_t D) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);

    int64_t b = idx / (S * H * D);
    int64_t idx1  = idx % (S*H*D);

    int64_t s = idx1 / (H*D);
    int64_t idx2 = idx1 % (H * D);
    // int64_t idx2 = idx1 % (2*2);
    int64_t d = idx2 % D;
    int64_t cos_idx = s * D + d;

    idx = idx * 2; // A single thread works on 2 elements

    if (idx >= N * 2) return;

    // Load bfloat16 values and convert to float for computation
    float x1_val = __bfloat162float(x[idx]);
    float x2_val = __bfloat162float(x[idx + 1]);
    float cos_val = __bfloat162float(cos[cos_idx]);
    float sin_val = __bfloat162float(sin[cos_idx]);

    // printf("idx: %lld, idx1: %lld, cos_idx: %lld, x1: %f, x2: %f, cos: %f, sin: %f\n", idx, idx1, cos_idx, x1_val, x2_val, cos_val, sin_val);

    float out1_val, out2_val;

    if (!conj) {
    out1_val = x1_val * cos_val - x2_val * sin_val;
    out2_val = x1_val * sin_val + x2_val * cos_val;
    } else {
    out1_val = x1_val * cos_val + x2_val * sin_val;
    out2_val = -x1_val * sin_val + x2_val * cos_val;
    }

    // Convert results back to bfloat16 and store
    out[idx] = __float2bfloat16(out1_val);
    out[idx + 1] = __float2bfloat16(out2_val);
}



// /*
__global__ void rotary_kernel(const float* x,
                            const float* cos, const float* sin,
                            float* out,
                            const int64_t N, const bool conj, const int32_t S, const int32_t H, const int32_t D) {
    // Kernel implementation
    /*
    for (int index = 0; index < work_per_thread; ++index) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + index * gridDim.x * blockDim.x;
        int64_t idx1  = idx % (1024*12*32);
        int64_t s = idx1 / (12*32);
        int64_t idx2 = idx1 % (12*32);
        int64_t d = idx2 / 32;
        int64_t cos_idx = s * 32 + d;

        if (idx >= N) return;

        float x1_val = x1[idx];
        float x2_val = x2[idx];
        float cos_val = cos[cos_idx];
        float sin_val = sin[cos_idx];
        float out1_val, out2_val;

        if (!conj) {
            out1_val = x1_val * cos_val - x2_val * sin_val;
            out2_val = x1_val * sin_val + x2_val * cos_val;
        } else {
            out1_val = x1_val * cos_val + x2_val * sin_val;
            out2_val = -x1_val * sin_val + x2_val * cos_val;
        }
        out1[idx] = out1_val;
        out2[idx] = out2_val;
    }
                                */
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    // printf("idx: %lld\n", idx);

    int64_t b = idx / (S * H * D);
    int64_t idx1  = idx % (S*H*D);

    int64_t s = idx1 / (H*D);
    int64_t idx2 = idx1 % (H * D);
    // int64_t idx2 = idx1 % (2*2);
    int64_t d = idx2 % D;
    int64_t cos_idx = s * D + d;

    idx = idx*2; // a single thread works on 2 elements

    // print id

    if (idx >= N*2) return;

    // printf("idx: %lld, idx1: %lld, cos_idx: %lld, x1: %f, x2: %f, cos: %f, sin: %f\n", idx, idx1, cos_idx, x[idx], x[idx+1], cos[cos_idx], sin[cos_idx]);

    float x1_val = x[idx];
    float x2_val = x[idx+1];
    float cos_val = cos[cos_idx];
    float sin_val = sin[cos_idx];
    float out1_val, out2_val;

    if (!conj) {
        out1_val = x1_val * cos_val - x2_val * sin_val;
        out2_val = x1_val * sin_val + x2_val * cos_val;
    } else {
        out1_val = x1_val * cos_val + x2_val * sin_val;
        out2_val = -x1_val * sin_val + x2_val * cos_val;
    }
    out[idx] = out1_val;
    out[idx+1] = out2_val;
}

void apply_rotary_cuda2(const torch::Tensor x,
                        const torch::Tensor cos, const torch::Tensor sin,
                        torch::Tensor out,
                        const bool conj)
{
    // auto iter = at::TensorIteratorConfig()
    // .add_output(out)
    // .add_input(x)
    // .add_input(cos)
    // .add_input(sin)
    // .check_all_same_dtype(false)
    // .promote_inputs_to_common_dtype(false)
    // .build();

    const int num_inputs = 3;
    const int num_outputs = 1;
    // if (iter.is_contiguous()) {
    //     auto input_calc = TrivialOffsetCalculator<num_inputs>();
    //     auto output_calc = TrivialOffsetCalculator<num_outputs>();
    //     std::cout << "Tensor is contiguous" << std::endl;
    // } else {
    //     auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    //     auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    //     std::cout << "Tensor is not contiguous" << std::endl;
    // }

    // check if x1 is contiguous
    // if (x.is_contiguous()) {
    //     std::cout << "x1 is contiguous" << std::endl;
    //     std::cout << x << std::endl;
    // } else {
    //     std::cout << "x1 is not contiguous" << std::endl;
    // }

    // // // // print dimensions of x1 and cos
    // // std::cout << "x1 dimensions: " << x1.sizes() << std::endl;
    // // std::cout << "cos dimensions: " << cos.sizes() << std::endl;
    // // std::cout << "out1 dimensions: " << out1.sizes() << std::endl;

    // // // print datatype of x1, out1, and cos
    // // std::cout << "x1 datatype: " << x1.dtype() << std::endl;
    // // std::cout << "out1 datatype: " << out1.dtype() << std::endl;
    // // std::cout << "cos datatype: " << cos.dtype() << std::endl;

    int64_t numel = x.numel();
    int64_t numel_cos = cos.numel();
    int64_t N = numel / 2;
    // std::cout << "N: " << N << std::endl;
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    auto num_threads_ = num_threads();
    int64_t grid = (N + num_threads_ - 1) / num_threads_;
    auto total_threads = grid * num_threads_;

    int64_t work_per_thread = (N + total_threads - 1) / total_threads;

    // std::cout << "N: " << N << std::endl;
    // std::cout << "grid: " << grid << std::endl;
    // std::cout << "num_threads_: " << num_threads_ << std::endl;
    // std::cout << "work_per_thread: " << work_per_thread << std::endl;

    // std::cout << "Grid size: " << grid << std::endl;
    // // // // std::cout << "Block size: " << block_work_size() << std::endl;
    // // // std::cout << "Number of elements: " << numel << std::endl;
    // std::cout << "Number of elements in cos: " << numel_cos << std::endl;
    // std::cout << "num_threads: " << num_threads() << std::endl;
    auto stream = at::cuda::getCurrentCUDAStream();

    // If x has [bz, S, H, D] shape then get S, H, D
    const int32_t S = (int32_t) x.size(1);
    const int32_t H = (int32_t) x.size(2);
    const int32_t D = (int32_t) x.size(3);

    // std::cout << x << std::endl;
    // std::cout << cos << std::endl;
    // std::cout << sin << std::endl;
    // std::cout << out << std::endl;

    // // Launch the kernel
    if (x.dtype() == at::kBFloat16) {
    //     // std::cout << "Launching kernel for bfloat16" << std::endl;
        rotary_kernel_bfloat16<<<grid, num_threads_, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), 
            reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()), 
            reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), 
            N, conj, S, H, D/2);
    } else {
        // std::cout << "Launching kernel for float" << std::endl;
        rotary_kernel<<<grid, num_threads_, 0, stream>>>(
            x.data_ptr<float>(), 
            cos.data_ptr<float>(), sin.data_ptr<float>(),
            out.data_ptr<float>(), 
            N, conj, S, H, D/2);
    }


    // std::cout << "CUDA kernel launched successfully" << std::endl;
}

// */