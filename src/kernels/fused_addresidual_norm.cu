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
    int warpnum = (blockDim.x + 31) / 32; // Fixed warp number calculation
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
                                        /*optional*/const T* bias,
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
    Vec_t* de_out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);

    T thread_accm = static_cast<T>(0);
    if (residual != nullptr){
        rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);
    }
     if (bias != nullptr){
        bia = reinterpret_cast<Vec_t*>(const_cast<T*>(bias));
    }

    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        if (residual != nullptr) {
            de_out[i].x += rsd[i].x;
            de_out[i].y += rsd[i].y;
            de_out[i].z += rsd[i].z;
            de_out[i].w += rsd[i].w;
            // Defect 1: Missing residual update (rsd[i] = de_out[i])
        }
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
        de_out[i].x = s[i].x * de_out[i].x * inv_fenmu;
        de_out[i].y = s[i].y * de_out[i].y * inv_fenmu;
        de_out[i].z = s[i].z * de_out[i].z * inv_fenmu;
        de_out[i].w = s[i].w * de_out[i].w * inv_fenmu;
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

    // Defect 2: Incorrect pointer arithmetic for vectorized access
    // Accessing like array[index] after casting to Vec_t* assumes byte stride, not Vec_t stride.
    Vec_t* de_out_vec = reinterpret_cast<Vec_t*>(decoder_out);
    Vec_t* rsd_vec = reinterpret_cast<Vec_t*>(residual);
    int row_offset_vec = batch_id * hidden_units / vec_size; // This offset logic is correct

    if (residual != nullptr && bias != nullptr){
         // rsd pointer potentially set but not used correctly below if rsd_vec is used
        bia = reinterpret_cast<Vec_t*>(const_cast<half*>(bias));
    }

    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        // Defect 2: Reading using incorrect indexing scheme
        dout = de_out_vec[row_offset_vec + i]; // Incorrect: should use base pointer + offset

        // Using separate pointers (potentially correct if initialized correctly)
        Vec_t* current_de_out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);
        Vec_t* current_rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);

        // Let's assume bias and scale pointers are correct relative to hidden_units start
        if(bias) bia = reinterpret_cast<Vec_t*>(const_cast<half*>(bias));
        if(scale) s = reinterpret_cast<Vec_t*>(const_cast<half*>(scale));


        // This section simulates trying to use the incorrectly accessed `dout`
        // or potentially mixing correct/incorrect pointers
        tmp = dout; // tmp holds potentially garbage data from wrong read
        if(residual) tmp = __hadd2(tmp, current_rsd[i]); // Adding correct residual to wrong initial value
        if(bias) tmp = __hadd2(tmp, bia[i]); // Adding correct bias

        // Defect 2: Writing back using incorrect index or pointer
        // de_out_vec[row_offset_vec + i] = tmp; // This would write to the wrong place

        // Let's simulate writing to the *correct* place but with corrupted data (tmp)
        current_de_out[i] = tmp;


        thread_accm += __half2float(tmp.x) * __half2float(tmp.x) + __half2float(tmp.y) * __half2float(tmp.y);
    }

    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ Vec_t inv_fenmu; // Should be float or scalar half based on usage
     if(tid == 0){
        // Using float for intermediate precision
        float inv_fenmu_float = rsqrt(blocksum / hidden_units + eps);
        inv_fenmu = scalar_cast_vec<Vec_t>(__float2half(inv_fenmu_float));
    }
    __syncthreads(); // Sync needed here

    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units); // Correct pointer for final output write
    if (scale != nullptr){
       s = reinterpret_cast<Vec_t*>(const_cast<half*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
         // Read the value that was just written (potentially corrupted)
         Vec_t current_val = out[i];
         // Apply scaling
         out[i] = __hmul2(__hmul2(s[i], current_val), inv_fenmu); // Final write using correct pointer 'out'
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
    // Improved block size calculation, assumes divisibility
    int num_threads = hidden_units / vec_size > 0 ? hidden_units / vec_size : 1;
    // Clamp to a max reasonable block size
    num_threads = num_threads > 1024 ? 1024 : num_threads;
    // Ensure at least warp size if possible
    num_threads = num_threads < 32 && hidden_units / vec_size > 0 ? 32 : num_threads;


    dim3 grid(batch_size);
    // Ensure num_threads is at least 1
    dim3 block(num_threads > 0 ? num_threads : 1);

    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>(residual ? residual->data : nullptr,
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