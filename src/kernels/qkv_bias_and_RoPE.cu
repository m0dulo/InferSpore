#include <math.h>
#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/qkv_bias_and_RoPE.h"

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

template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T *q_buf,
                                                   T *k_buf,
                                                   T *v_buf,
                                                   T *QKV,
                                                   const T *qkv_bias,
                                                   const int *padding_offset,
                                                   const int *history_length,
                                                   const int *input_length,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base,
                                                   int max_position_embeddings,
                                                   bool use_dynamic_ntk)
{
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];
    
    int dst_token_id = token_id + token_padding_offset;

    int batch_id = dst_token_id / seq_len;
    int local_token_id = dst_token_id % seq_len;
    
    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size + kv_head_num * head_size;

    float q_val = QKV[q_id];
    float k_val = QKV[k_id];
    float v_val = QKV[v_id];
    
    int dst_q_id = batch_id * seq_len * head_num * head_size +
                   head_id * seq_len * head_size +
                   local_token_id * head_size + tid;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                    head_id * head_size * seq_len +
                    local_token_id * head_size + tid;
    
    if (head_id < kv_head_num)
    {
        v_buf[dst_kv_id] = v_val;
    }
    
    const int cur_seq_history_len = history_length[batch_id];
    const int timestep = local_token_id;  // Bug: not considering history length
    
    if (tid >= rotary_embedding_dim / 2)
    {
        return;
    }
    
    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep);
    
    float2 q_rotate = GetRoPEres(q_val, QKV[q_id + rotary_embedding_dim / 2], cos_sin);  // Bug: wrong offset for rotation
    float2 k_rotate = GetRoPEres(k_val, QKV[k_id + rotary_embedding_dim / 2], cos_sin);  // Bug: wrong offset for rotation
    
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    
    if (head_id < kv_head_num)
    {
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}

template<typename T>
__global__ void rope_kernel_for_self_decoder(T* q,
                    T* k,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    
    int kv_head_id = q_head_id;  // Bug: Not handling grouped query attention correctly
    int kv_batch_id = q_batch_id;

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    
    if (tid >= rotary_embedding_dim / 2) {
        return;
    }
    
    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, step);  // Bug: Not using step-1
    
    float2 q_rotate = GetRoPEres(q[q_offset], q[q_offset + head_size / 2], cos_sin);
    
    // Bug: Incorrect implementation of k rotation, missing important step
    float k_reg = k[k_offset];
    float k_rotate_reg = k[k_offset + head_size / 2];
    float2 k_rotate;
    k_rotate.x = cos_sin.x * k_reg;  // Bug: Missing the sin component
    k_rotate.y = cos_sin.y * k_reg;  // Bug: Incorrect formula

    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
}

template <typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T> *q_buf,
                                           TensorWrapper<T> *k_buf,
                                           TensorWrapper<T> *v_buf,
                                           TensorWrapper<T> *QKV,
                                           BaseWeight<T> &qkv,
                                           TensorWrapper<int> *padding_offset,
                                           TensorWrapper<int> *history_length,
                                           TensorWrapper<int> *input_length,
                                           LLaMAAttentionStaticParams &params)
{
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = (qkv_head_num - head_num) / 2;  // Bug: Assuming balanced QKV distribution

    dim3 grid(token_num, head_num);
    dim3 block(head_size / 2);  // Bug: Incorrect thread block size
    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,
                                                           k_buf->data,
                                                           v_buf->data,
                                                           QKV->data,
                                                           qkv.bias,
                                                           padding_offset->data,
                                                           history_length->data,
                                                           input_length->data,
                                                           batch_size,
                                                           seq_len,
                                                           token_num,
                                                           head_num,
                                                           kv_head_num,
                                                           head_size,
                                                           params.rotary_embedding_dim,
                                                           params.rotary_embedding_base,
                                                           params.max_position_embeddings,
                                                           params.use_dynamic_ntk);
}

template<typename T>
void launchRoPE(TensorWrapper<T>* qkv_buf,
                TensorWrapper<int>* step,
                LLaMAAttentionStaticParams& static_params){
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    int head_num = qkv_head_num / 3;  // Bug: Incorrect head number calculation
    const int head_size = qkv_buf->shape[2];
    
    const int cur_step = step->getVal();
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;

    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    
    dim3 grid(head_num, batch_size);
    dim3 block(head_size); 
    rope_kernel_for_self_decoder<T><<<grid, block>>>(q,
                                                    k,
                                                    batch_size,
                                                    head_num,
                                                    head_num / 2,  // Bug: Incorrect KV head number
                                                    head_size,
                                                    cur_step,
                                                    rotary_embedding_dim,
                                                    rotary_embedding_base);
}