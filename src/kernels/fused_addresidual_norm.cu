#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_addresidual_norm.h"

template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = blockDim.x / 32; 
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warpsum[tid] : (T)0.0f;
    sum = warpReduceSum<T>(sum);
    return sum;
}

template<typename T>
__global__ void FusedAddBiasResidualRMSNorm(
                                        T* residual,
                                        T* decoder_out,
                                        const T* bias,
                                        const T* scale,
                                        float eps,
                                        int num_tokens,
                                        int hidden_units){
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *rsd, *bia, *s;
    Vec_t tmp;
    Vec_t* de_out = reinterpret_cast<Vec_t*>(decoder_out);

    T thread_accm = static_cast<T>(0);
    rsd = reinterpret_cast<Vec_t*>(residual);
    if (bias != nullptr){
        bia = reinterpret_cast<Vec_t*>(const_cast<T*>(bias));
    }

    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        int linear_idx = batch_id * (hidden_units / vec_size) + i; 
        if (residual != nullptr) {
            de_out[linear_idx].x += rsd[linear_idx].x;
            de_out[linear_idx].y += rsd[linear_idx].y;
            de_out[linear_idx].z += rsd[linear_idx].z;
            de_out[linear_idx].w += rsd[linear_idx].w;
        }
        if (bias != nullptr) {
            de_out[linear_idx].x += bia[i].x;
            de_out[linear_idx].y += bia[i].y;
            de_out[linear_idx].z += bia[i].z;
            de_out[linear_idx].w += bia[i].w;
        }
        thread_accm += de_out[linear_idx].x * de_out[linear_idx].x;
        thread_accm += de_out[linear_idx].y * de_out[linear_idx].y;
        thread_accm += de_out[linear_idx].z * de_out[linear_idx].z;
        thread_accm += de_out[linear_idx].w * de_out[linear_idx].w;
    }

    T blocksum = blockReduceSum<T>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrt(blocksum / hidden_units + eps);
    }
    __syncthreads();

    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        int linear_idx = batch_id * (hidden_units / vec_size) + i; 
        de_out[linear_idx].x = s[i].x * de_out[linear_idx].x * inv_fenmu;
        de_out[linear_idx].y = s[i].y * de_out[linear_idx].y * inv_fenmu;
        de_out[linear_idx].z = s[i].z * de_out[linear_idx].z * inv_fenmu;
        de_out[linear_idx].w = s[i].w * de_out[linear_idx].w * inv_fenmu;
    }
}

template<>
__global__ void FusedAddBiasResidualRMSNorm(
                                        half* residual,
                                        half* decoder_out,
                                        const half* bias,
                                        const half* scale,
                                        float eps,
                                        int num_tokens,
                                        int hidden_units){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *rsd, *bia, *s;
    Vec_t dout, tmp;
    float thread_accm = 0.0f;
    if (residual != nullptr && bias != nullptr){
        rsd = reinterpret_cast<Vec_t*>(residual);
        bia = reinterpret_cast<Vec_t*>(const_cast<half*>(bias));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        int linear_idx = batch_id * (hidden_units / vec_size) + i; 
        dout = reinterpret_cast<Vec_t*>(decoder_out)[linear_idx];
        tmp = __hadd2(__hadd2(dout, rsd[linear_idx]), bia[i]);
        reinterpret_cast<Vec_t*>(decoder_out)[linear_idx] = tmp;
        thread_accm += __half2float(tmp.x) * __half2float(tmp.x) + __half2float(tmp.y) * __half2float(tmp.y);
    }

    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ Vec_t inv_fenmu;
    if(tid == 0){
        inv_fenmu = scalar_cast_vec<Vec_t>(__float2half(rsqrt(blocksum / hidden_units + eps)));
    }
    __syncthreads();

    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_out);
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<half*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
         int linear_idx = batch_id * (hidden_units / vec_size) + i; 
        out[linear_idx] = __hmul2(__hmul2(s[i], out[linear_idx]), inv_fenmu);
    }
}


template<typename T>
void launchFusedAddBiasResidualRMSNorm(
                                        TensorWrapper<T>* residual,
                                        TensorWrapper<T>* decoder_out,
                                        BaseWeight<T>& norm,
                                        T* scale,
                                        float eps)
{
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    T* bias = norm.bias;
    T* gamma = scale;
    int vec_size = Vec<T>::size;
    int num_threads = (hidden_units / vec_size + 31) / 32 * 32; 
    num_threads = (num_threads == 0) ? 32 : num_threads; 
    num_threads = (num_threads > 1024) ? 1024 : num_threads; 

    dim3 grid(batch_size);
    dim3 block(num_threads);
    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>(residual->data,
                                               decoder_out->data,
                                               bias,
                                               gamma,
                                               eps,
                                               batch_size,
                                               hidden_units);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(decoder_out->data);
#else
#endif
}
template void launchFusedAddBiasResidualRMSNorm(
                                        TensorWrapper<float>* residual,
                                        TensorWrapper<float>* decoder_out,
                                        BaseWeight<float>& norm,
                                        float* scale,
                                        float eps);
template void launchFusedAddBiasResidualRMSNorm(
                                        TensorWrapper<half>* residual,
                                        TensorWrapper<half>* decoder_out,
                                        BaseWeight<half>& norm,
                                        half* scale,
                                        float eps);