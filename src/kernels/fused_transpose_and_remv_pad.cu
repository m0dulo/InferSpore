#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_decoder_self_attention.h"

template<typename T>
__device__ T warpReduceSum(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);
}

template<typename T>
__device__ T warpReduceMax(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpmax[64];
    val = warpReduceMax(val);
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}

inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}

template<typename T>
__global__ void masked_MHA_kernel(T* q,
                    T* k,
                    T* v,
                    T* qkv_bias,
                    T* k_cache,
                    T* v_cache,
                    T* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int q_head_id = bid % head_num;
    int q_batch_id = bid / head_num;
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    int cache_offset = batch_size * kv_batch_stride;
    float scale = rsqrt(float(head_size));

    extern __shared__ char sqk[];
    T* sq = reinterpret_cast<T*>(sqk);
    T* sk = sq + head_size;
    float* logits = reinterpret_cast<float*>(sk + head_size);

    sq[tid] = q[q_offset];
    if (qkv_bias != nullptr){
        sq[tid] += qkv_bias[q_head_id * head_size + tid];
    }
    __syncthreads();

    for(int iter = 0; iter < step; iter++) {
        sk[tid] = k_cache[iter * cache_offset + k_offset];
        if (iter == step - 1) {
            sk[tid] = k[k_offset];
            k_cache[iter * cache_offset + k_offset] = k[k_offset];
        }

        T qk = sq[tid] * sk[tid] * scale;
        T attn_score = blockReduceSum<T>(qk);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }

    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;
    
    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();
    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();
    if(tid < step) {
        logits[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();

    T O = 0.0f;
    for(int iter = 0; iter < step; iter++) {
        T value = v_cache[iter * cache_offset + k_offset];
        if (iter == step - 1) {
            value = v[k_offset];
            v_cache[iter * cache_offset + k_offset] = v[k_offset];
        }
        O += value * logits[iter];
    }
    mha_output[q_offset] = O;
}

template<>
__global__ void masked_MHA_kernel(half* q,
                    half* k,
                    half* v,
                    half* qkv_bias,
                    half* k_cache,
                    half* v_cache,
                    half* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int q_head_id = bid % head_num;
    int q_batch_id = bid / head_num;
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    int cache_offset = batch_size * kv_batch_stride;
    half scale = __float2half(rsqrt(float(head_size)));
 
    extern __shared__ char sqk[];
    half* sq = reinterpret_cast<half*>(sqk);
    half* sk = sq + head_size;
    float* logits = reinterpret_cast<float*>(sk + head_size);

    sq[tid] = q[q_offset];
    if (qkv_bias != nullptr){
        sq[tid] = __hadd(sq[tid], qkv_bias[q_head_id * head_size + tid]);
    }
    __syncthreads();

    for(int iter = 0; iter < step; iter++) {
        sk[tid] = k_cache[iter * cache_offset + k_offset];
        if (iter == step - 1) {
            sk[tid] = k[k_offset];
            k_cache[iter * cache_offset + k_offset] = k[k_offset];
        }

        half qk = __hmul(__hmul(sq[tid], sk[tid]), scale);
        float qk_fp32 = __half2float(qk);
        float attn_score = blockReduceSum<float>(qk_fp32);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }

    float local_logits = tid < step ? logits[tid] : 0;
    __shared__ float row_max, fenmu;
    
    float block_max = blockReduceMax<float>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();
    float fenzi = tid < step ? expf(local_logits - row_max) : 0;
    
    float block_fenmu = blockReduceSum<float>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();
    if(tid < step) {
        logits[tid] = (float)(fenzi / fenmu);
    }
    __syncthreads();

    float O = 0.0f;
    for(int iter = 0; iter < step; iter++) {
        half value = v_cache[iter * cache_offset + k_offset];
        if (iter == step - 1) {
            value = v[k_offset];
            v_cache[iter * cache_offset + k_offset] = v[k_offset];
        }
        O += __half2float(value) * logits[iter];
    }
    mha_output[q_offset] = __float2half(O);
}

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,
                            BaseWeight<T>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<T>* mha_output,
                            LLaMAAttentionStaticParams& static_params){
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3];
    int head_num = qkv_head_num - 2 * kv_head_num;
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;
    size_t smem_size_bytes = 2 * head_size * sizeof(T) + cur_step * sizeof(float);
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;
    T* v = qkv_data + (head_num + kv_head_num) * head_size;

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;
    bool  use_dynamic_ntk = static_params.use_dynamic_ntk;
    dim3 grid(head_num * batch_size);
    dim3 block(head_size);
    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(q,
                                                            k,
                                                            v,
                                                            qkv.bias,
                                                            k_cache->data + layer_offset,
                                                            v_cache->data + layer_offset,
                                                            mha_output->data,
                                                            batch_size,
                                                            head_num,
                                                            kv_head_num,
                                                            max_seq_len,
                                                            head_size,
                                                            cur_step,
                                                            rotary_embedding_dim,
                                                            rotary_embedding_base);
}

template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf,
                            BaseWeight<float>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<float>* k_cache,
                            TensorWrapper<float>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<float>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf,
                            BaseWeight<half>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<half>* k_cache,
                            TensorWrapper<half>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<half>* mha_output,
                            LLaMAAttentionStaticParams& static_params);