#include <iostream>
#include <math.h>
#include <cuda_fp16.h>
#include "src/kernels/act_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
__device__ __forceinline__ T silu(const T& in) {
  return (T) (((float) in) / (1.0f + expf((float) -in)));
}

template<>
__device__ __forceinline__ half silu<half>(const half& in) {
  float in_f = __half2float(in);
  float result_f = in_f / (1.0f + expf(-in_f));
  return __float2half(result_f);
}

template<>
__device__ __forceinline__ half2 silu<half2>(const half2& in) {
    return make_half2(__float2half(silu<float>((float)(in.x))), __float2half(silu<float>((float)(in.y))));
}


template<typename T>
__global__ void silu_and_mul_kernel(
  T* out,               // shape: [bs, intermedia size]
  const T* input,       // shape: [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
    const int input_base_idx = batch_idx * 2 * intermedia_size;
    const int output_base_idx = batch_idx * intermedia_size; 
    const T x = input[input_base_idx + idx];
    const T y = input[input_base_idx + intermedia_size + idx];
    out[output_base_idx + idx] = silu<T>(x) * y;
  }
}

template<>
__global__ void silu_and_mul_kernel<half>(
  half* out,               
  const half* input,       
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  int vec_size = Vec<half>::size;
  using Vec_t = typename Vec<half>::Type;
  for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x * vec_size) {
    const int input_base_idx = batch_idx * 2 * intermedia_size;
    const int output_base_idx = batch_idx * intermedia_size;
    const Vec_t x = *reinterpret_cast<const Vec_t*>(&input[input_base_idx + idx]);
    const Vec_t y = *reinterpret_cast<const Vec_t*>(&input[input_base_idx + intermedia_size + idx]);
   
    *reinterpret_cast<Vec_t*>(&out[output_base_idx + idx]) = __hmul2(silu<Vec_t>(x), y);
  }
}


template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    if (getTensorType<T>() == DataType::FP16) {
         LLM_CHECK(intermedia_size % Vec<half>::size == 0);
    }

    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#else
#endif
}

template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* output);