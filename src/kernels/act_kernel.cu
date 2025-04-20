#include <iostream>
#include "src/kernels/act_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"
// fp32 silu version
template<typename T>
__device__ __forceinline__ T silu(const T& in) {
  // x * sigmoid(x)
  return (T) (((float) in) / (1.0f + expf((float) -in)));
}
// fp16 silu version
template<>
__device__ __forceinline__ half2 silu<half2>(const half2& in) {
  return make_half2(__float2half(silu<float>((float)(in.x))), __float2half(silu<float>((float)(in.y))));
}

// 代码逻辑：第一个intermediate 去做silu，其结果与第二个intermediate 做点乘mul
template<typename T>
__global__ void silu_and_mul_kernel(
  T* out,               // shape: [bs, intermedia size]
  const T* input,       // shape: [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  // 循环处理intermedia_size，当线程不够时，使得也能处理完
  for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) { 
    // 第一个和第二个intermediate处于同一buffer: input
    // 根据shape索引第一个intermediate
    const T x = input[batch_idx * 2 * intermedia_size + idx];
    // 根据shape索引第二个intermediate
    const T y = input[batch_idx * 2 * intermedia_size + intermedia_size + idx];
    // 索引到了后做计算，把计算结果写回output
    out[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
  }
}

template<>
__global__ void silu_and_mul_kernel<half>(
  half* out,               // [bs, intermedia size]
  const half* input,       // [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  // 获取fp16的向量大小
  int vec_size = Vec<half>::size;
  // 获取fp16的向量类型half2
  using Vec_t = typename Vec<half>::Type;
  for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
    // 与fp32实现的不同在于
    // 1.向量化读取
    // 2.使用hmul2向量化计算
    // 3.向量化写入
    const Vec_t x = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + idx]));
    const Vec_t y = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + intermedia_size + idx]));
    *reinterpret_cast<Vec_t*>(&out[batch_idx * intermedia_size + idx]) = __hmul2(silu<Vec_t>(x), y);
  }
}

template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    int batch_size = input->shape[0];
    // 预防性检查，主要是防止shape的定义写错，导致不是我们expect的，那就比较难debug了
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
    // for debug，打印swiglu这个kernel的输出结果
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#else
#endif
}
// We must instancite the template, if not, will report linking issue
template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* output);
