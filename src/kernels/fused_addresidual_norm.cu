#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_addresidual_norm.h"

template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}
// (m0dulo) notes:!!!when blocksize < 32, use blockDim.x/32 to get warp nums is wrong, we should ceil it instead
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warpsum[tid] : (T)0.0f;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    return sum;
}
// 1.this kernel is used after self attention in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
template<typename T>
__global__ void FusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    T* residual, 
                                    T* decoder_out, // [num tokens, hidden_units]
                                    /*optional*/const T* bias,  // [hidden_units]
                                    const T* scale, // [hidden_units], RMSNorm weights
                                    float eps, // RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units){
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *rsd, *bia, *s;
    Vec_t tmp;
    Vec_t* de_out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);// note the offset should divide vec size

    T thread_accm = static_cast<T>(0);
    rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);//note the offset     should divide vec size
    if (bias != nullptr){
        bia = reinterpret_cast<Vec_t*>(const_cast<T*>(bias));
    } 

    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        if (residual != nullptr) {
            de_out[i].x += rsd[i].x;
            de_out[i].y += rsd[i].y;
            de_out[i].z += rsd[i].z;
            de_out[i].w += rsd[i].w;
            // update residual to be used in add residual kernel at the end of every decoder layer
            rsd[i].x = de_out[i].x;
            rsd[i].y = de_out[i].y;
            rsd[i].z = de_out[i].z;
            rsd[i].w = de_out[i].w;
        }
        //TODO: to update rsd by rsd + bias when bias is valid
        if (bias != nullptr) {
            de_out[i].x += bia[i].x;
            de_out[i].y += bia[i].y;
            de_out[i].z += bia[i].z;
            de_out[i].w += bia[i].w;
        }	    
        thread_accm += de_out[i].x * de_out[i].x;
        thread_accm += de_out[i].y * de_out[i].y;
        thread_accm += de_out[i].z * de_out[i].z;
        thread_accm += de_out[i].w * de_out[i].w;
    } // addresidual
    // mean(x^2)
    T blocksum = blockReduceSum<T>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
	    inv_fenmu = rsqrt(blocksum / hidden_units + eps);
        //debug info printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    __syncthreads();
    // rmsnorm
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        //s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale))[i];
        de_out[i].x = s[i].x * de_out[i].x * inv_fenmu;
        de_out[i].y = s[i].y * de_out[i].y * inv_fenmu;
        de_out[i].z = s[i].z * de_out[i].z * inv_fenmu;
        de_out[i].w = s[i].w * de_out[i].w * inv_fenmu;
    }
}

template<>
__global__ void FusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    half* residual, 
                                    half* decoder_out, // [num tokens, hidden_units]
                                    const half* bias, //[hidden_units]
                                    const half* scale, //[hidden_units], RMSNorm weights
                                    float eps, //RMSNorm eps
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
        rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);//note the offset     should divide vec size
        bia = reinterpret_cast<Vec_t*>(const_cast<half*>(bias));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        dout = reinterpret_cast<Vec_t*>(decoder_out)[batch_id * hidden_units / vec_size + i];// note the offset should divide vec size
        tmp = __hadd2(__hadd2(dout, rsd[i]), bia[i]);
        thread_accm += __half2float(tmp.x) * __half2float(tmp.x) + __half2float(tmp.y) * __half2float(tmp.y);
    } // addresidual
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ Vec_t inv_fenmu;
    if(tid == 0){
        //debug info printf("blocksum on GPU is %f\n", blocksum);
        inv_fenmu = scalar_cast_vec<Vec_t>(__float2half(rsqrt(blocksum / hidden_units + eps)));
        //debug info printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    // rmsnorm
    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);// note before vec the stride is batch_id * hiddenunits w/o / vecsize
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<half*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        out[i] = __hmul2(__hmul2(s[i], out[i]), inv_fenmu);
    } 
}

template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<T>* residual, 
                                    TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<T>& norm,
                                    T* scale, //RMSNorm weights
                                    float eps) //RMSNorm eps
{
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    T* bias = norm.bias;
    T* gamma = scale;
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / vec_size; // assume head size can be divided by 4 and 2
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
template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<float>* residual, 
                                    TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<float>& norm,
                                    float* scale, //RMSNorm weights
                                    float eps);
template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<half>* residual, 
                                    TensorWrapper<half>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<half>& norm,
                                    half* scale, //RMSNorm weights
                                    float eps);