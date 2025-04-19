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
  T* out,
  const T* input,
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
    const int base_idx = batch_idx * 2 * intermedia_size;
    const T x = input[base_idx + idx];
    const T y = input[base_idx + intermedia_size + idx];
    out[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
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
    const int base_idx = batch_idx * 2 * intermedia_size;

    const Vec_t x = *reinterpret_cast<const Vec_t*>(&input[base_idx + idx]);
    const Vec_t y = *reinterpret_cast<const Vec_t*>(&input[base_idx + intermedia_size + idx]);
    *reinterpret_cast<Vec_t*>(const_cast<half*>(&out[batch_idx * intermedia_size + idx])) = __hmul2(silu<Vec_t>(x), y);
  }
}


template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    LLM_CHECK(getTensorType<T>() != DataType::FP16 || intermedia_size % Vec<half>::size == 0);

    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#else
#endif
}

template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(Wrapper<half>* input, TensorWrapper<half>* output);