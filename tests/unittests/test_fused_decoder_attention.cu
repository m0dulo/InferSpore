#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cassert>

#include "src/kernels/fused_decoder_self_attention.h"
#include "src/utils/macro.h"

template <typename T>
void CPUMaskedAttn(T *q,
                   T *k,
                   T *v,
                   T *k_cache,
                   T *v_cache,
                   float *mha_output,
                   const int batch_size,
                   const int num_heads,
                   const int kv_num_heads,
                   const int head_size,
                   int step,
                   int rotary_embedding_dim = 0,
                   float rotary_embedding_base = 10000.0f)
{
    int batch_stride = num_heads * head_size;
    int kv_batch_stride = kv_num_heads * head_size;
    int head_stride = head_size;
    int block_nums = batch_size * num_heads;
    float scale = rsqrt(float(head_size));

    const T *q_mem = q;
    const T *k_mem = k;
    const T *v_mem = v;

    float *sqk = (float *)malloc(sizeof(float) * (block_nums * (3 * head_size + step)));
    float *sq = sqk;
    float *sk = sq + block_nums * head_size;
    float *logits = sk + block_nums * head_size;
    float *sv = logits + block_nums * step;
    for (int batch_id = 0; batch_id < batch_size; batch_id++)
    {
        for (int head_id = 0; head_id < num_heads; head_id++)
        {
            int kv_head_id = head_id / (num_heads / kv_num_heads);
            
            float row_max = -INFINITY;
            for (int iter = 0; iter < step; iter++)
            {
                float attn_score = 0.0f;
                for (int tid = 0; tid < head_size; tid++)
                {
                    int q_offset = batch_id * batch_stride + head_id * head_stride + tid;
                    int k_offset = batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
                    int cache_offset = batch_id * kv_num_heads * step * head_size 
                                    + kv_head_id * step * head_size
                                    + iter * head_size + tid;
                    
                    sk[tid] = (float)k_cache[cache_offset];
                    if (iter == step - 1)
                    {
                        k_cache[cache_offset] = k_mem[k_offset];
                        sk[tid] = (float)k_mem[k_offset];
                        
                        // Approximate RoPE application for k
                        if (rotary_embedding_dim > 0 && tid < rotary_embedding_dim / 2) {
                            int pair_id = tid * 2;
                            float inv_freq = (float)step / pow(rotary_embedding_base, pair_id / (float)rotary_embedding_dim);
                            float cos_val = cos(inv_freq);
                            float sin_val = sin(inv_freq);
                            float k_val = (float)k_mem[k_offset];
                            float k_rot_val = (pair_id + 1 < head_size) ? (float)k_mem[k_offset + 1] : 0.0f;
                            sk[tid] = cos_val * k_val - sin_val * k_rot_val;
                            if (pair_id + 1 < head_size) {
                                sk[tid+1] = cos_val * k_rot_val + sin_val * k_val;
                            }
                        }
                    }

                    sq[tid] = (float)q_mem[q_offset];
                    
                    // Approximate RoPE application for q
                    if (rotary_embedding_dim > 0 && tid < rotary_embedding_dim / 2 && iter == step - 1) {
                        int pair_id = tid * 2;
                        float inv_freq = (float)step / pow(rotary_embedding_base, pair_id / (float)rotary_embedding_dim);
                        float cos_val = cos(inv_freq);
                        float sin_val = sin(inv_freq);
                        float q_val = (float)q_mem[q_offset];
                        float q_rot_val = (pair_id + 1 < head_size) ? (float)q_mem[q_offset + 1] : 0.0f;
                        sq[tid] = cos_val * q_val - sin_val * q_rot_val;
                        if (pair_id + 1 < head_size) {
                            sq[tid+1] = cos_val * q_rot_val + sin_val * q_val;
                        }
                    }
                    
                    float qk = sq[tid] * sk[tid] * scale;
                    attn_score += qk;
                }
                logits[batch_id * num_heads * step + head_id * step + iter] = attn_score;
                row_max = std::max(attn_score, row_max);
            }
            
            float fenmu = 0.0f;
            for (int iter = 0; iter < step; iter++)
            {
                float fenzi = expf(logits[batch_id * num_heads * step + head_id * step + iter] - row_max);
                fenmu += fenzi;
                logits[batch_id * num_heads * step + head_id * step + iter] = fenzi;
            }
            for (int iter = 0; iter < step; iter++)
            {
                logits[batch_id * num_heads * step + head_id * step + iter] = logits[batch_id * num_heads * step + head_id * step + iter] / fenmu;
            }
            
            for (int tid = 0; tid < head_size; tid++)
            {
                float O = 0.0f;
                int q_offset = batch_id * batch_stride + head_id * head_stride + tid;
                int k_offset = batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
                
                for (int iter = 0; iter < step; iter++)
                {
                    int cache_offset = batch_id * kv_num_heads * step * head_size 
                                     + kv_head_id * step * head_size
                                     + iter * head_size + tid;
                    
                    sv[tid] = (float)v_cache[cache_offset];
                    if (iter == step - 1)
                    {
                        v_cache[cache_offset] = v_mem[k_offset];
                        sv[tid] = (float)v_mem[k_offset];
                    }
                    O += sv[tid] * logits[batch_id * num_heads * step + head_id * step + iter];
                }
                mha_output[q_offset] = O;
            }
        }
    }

    free(sqk);
}

template <typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size)
{
    bool passed = true;
    int errors = 0;
    
    for (int i = 0; i < output_size; i++)
    {
        float GPUres = (float)GPUoutput[i];
        if (fabs(CPUoutput[i] - GPUres) > 1e-3)
        {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUres);
            errors++;
            passed = false;
            if (errors >= 10) {
                printf("Too many errors, stopping comparison\n");
                break;
            }
        }
    }
    return passed;
}

int main(int argc, char *argv[])
{
    constexpr int batch_size = 1;
    constexpr int head_size = 4;
    constexpr int num_heads = 2;
    constexpr int kv_num_heads = 2;
    constexpr int max_seq_len = 4;
    int h_step = 4;
    int h_layer_id = 0;
    int rotary_embedding_dim = 128;
    float rotary_embedding_base = 10000;
    int max_position_embeddings = 2048;
    bool use_dynamic_ntk = false;
    float *h_qkv;                                                                                                                                 
    float *d_qkv;                                                                                                                                 
    int qkv_size = batch_size * (2 * kv_num_heads + num_heads) * head_size;                                                                       
    h_qkv = (float *)malloc(sizeof(float) * qkv_size);                                                                                            
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_qkv, sizeof(float) * qkv_size));                                                                    
    float *h_kcache;                                                                                                                              
    float *d_kcache;                                                                                                                              
    int kcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_kcache = (float *)malloc(sizeof(float) * kcache_size);                                                                                      
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_kcache, sizeof(float) * kcache_size));                                                              
    float *h_vcache;                                                                                                                              
    float *d_vcache;                                                                                                                              
    int vcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_vcache = (float *)malloc(sizeof(float) * vcache_size);                                                                                      
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_vcache, sizeof(float) * vcache_size));                                                              
    for (int i = 0; i < qkv_size; i++)                                                                                                            
    {                                                                                                                                             
        if (i < batch_size * num_heads * head_size)                                                                                               
        {                                                                                                                                         
            if (i < batch_size * num_heads * head_size / 2)                                                                                       
            {                                                                                                                                     
                h_qkv[i] = (float)(i + 1);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (float)(i - 3) / 10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        else if (i < batch_size * (num_heads + kv_num_heads) * head_size)                                                                         
        {                                                                                                                                         
            if (i < batch_size * (num_heads + kv_num_heads / 2) * head_size)                                                                      
            {                                                                                                                                     
                h_qkv[i] = (float)(i + 5);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (float)(i + 1) / 10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        else if (i < batch_size * (num_heads + kv_num_heads * 2) * head_size)                                                                     
        {                                                                                                                                         
            if (i < batch_size * (num_heads + kv_num_heads + kv_num_heads / 2) * head_size)                                                       
            {                                                                                                                                     
                h_qkv[i] = (float)(i - 3);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (float)(i - 7) / 10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
    }                                                                                                                                             
    float *h_q = h_qkv;                                                                                                                           
    float *h_k = h_q + batch_size * num_heads * head_size;                                                                                        
    float *h_v = h_k + batch_size * (kv_num_heads + num_heads) * head_size;                                                                       
    for (int i = 0; i < (kcache_size * h_step) / max_seq_len; i++)                                                                                
    {                                                                                                                                             
        if (i < kcache_size / 2)                                                                                                                  
        {                                                                                                                                         
            h_kcache[i] = (float)(i + 1);                                                                                                         
            h_vcache[i] = (float)(i + 1);                                                                                                         
        }                                                                                                                                        
        else                                                                                                                                     
        {                                                                                                                                        
            h_kcache[i] = (float)(i - kcache_size / 2 + 1) / 10;                                                                                 
            h_vcache[i] = (float)(i - kcache_size / 2 + 1) / 10;                                                                                 
        }                                                                                                                                        
    }                                                                                                                                            
    float *h_o;                                                                                                                                  
    float *d_o;                                                                                                                                  
    int o_size = batch_size * num_heads * head_size;                                                                                             
    h_o = (float *)malloc(sizeof(float) * o_size);                                                                                               
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_o, sizeof(float) * o_size));                                                                        
    bool *h_finished = (bool *)malloc(sizeof(bool) * batch_size);                                                                                
    bool *d_finished;                                                                                                                            
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size));                                                              
    for (int i = 0; i < batch_size; i++)                                                                                                         
    {                                                                                                                                            
        h_finished[i] = static_cast<bool>(0);                                                                                                    
    }                                                                                                                                            
    float *h_qkv_bias = (float *)malloc(sizeof(float) * (2 * kv_num_heads + num_heads) * head_size);                                             
    float *d_qkv_bias;                                                                                                                           
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size));                            
    for (int i = 0; i < (2 * kv_num_heads + num_heads) * head_size; i++)                                                                         
    {                                                                                                                                            
        h_qkv_bias[i] = (float)0.0f;                                                                                                             
    }                                                                                                                                            
    CHECK_CUDA_ERROR(cudaMemcpy(d_qkv, h_qkv, sizeof(float) * batch_size * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice));  
    CHECK_CUDA_ERROR(cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice));     
    CHECK_CUDA_ERROR(cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));                                      
    CHECK_CUDA_ERROR(cudaMemcpy(d_kcache, h_kcache, sizeof(float) * kcache_size, cudaMemcpyHostToDevice));                                        
    CHECK_CUDA_ERROR(cudaMemcpy(d_vcache, h_vcache, sizeof(float) * vcache_size, cudaMemcpyHostToDevice));                                        
    DataType type = getTensorType<float>();                                                                                                      
    DataType type_bool = getTensorType<bool>();                                                                                                  
    DataType type_int = getTensorType<int>();                                                                                                    
    TensorWrapper<float> *qkv = new TensorWrapper<float>(GPU, type, {batch_size, num_heads + 2 * kv_num_heads, head_size}, d_qkv);               
    TensorWrapper<float> *kcache = new TensorWrapper<float>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_kcache);
    TensorWrapper<float> *vcache = new TensorWrapper<float>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_vcache);
    TensorWrapper<bool> *finished = new TensorWrapper<bool>(GPU, type_bool, {batch_size}, d_finished);                                           
    TensorWrapper<int> *step = new TensorWrapper<int>(CPU, type_int, {1}, &h_step);                                                              
    TensorWrapper<int> *layer_id = new TensorWrapper<int>(CPU, type_int, {1}, &h_layer_id);                                                      
    TensorWrapper<float> *mha_output = new TensorWrapper<float>(GPU, type, {batch_size, num_heads, head_size}, d_o);                             
    BaseWeight<float> qkv_weight;                                                                                                                
    qkv_weight.bias = d_qkv_bias;                                                                                                                
    LLaMAAttentionStaticParams params;                                                                                                           
    params.rotary_embedding_dim = rotary_embedding_dim;                                                                                          
    params.rotary_embedding_base = rotary_embedding_base;                                                                                        
    params.max_position_embeddings = max_position_embeddings;                                                                                    
    params.use_dynamic_ntk = false;                                                                                                              
    

    try {
        launchDecoderMaskedMHA(qkv, qkv_weight, layer_id, kcache, vcache, finished, step, mha_output, params);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        printf("GPU kernel execution failed: %s\n", e.what());
        return 1;
    }
    
 
    CHECK_CUDA_ERROR(cudaMemcpy(h_o, d_o, sizeof(float) * o_size, cudaMemcpyDeviceToHost));                                                      
    

    float *CPU_output = (float *)malloc(sizeof(float) * o_size);                                                                                 
    CPUMaskedAttn<float>(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, kv_num_heads, head_size, h_step, rotary_embedding_dim, rotary_embedding_base);                               
    
 
    bool is_true = CheckResult<float>(CPU_output, h_o, o_size);                                                                                  
    if (is_true)                                                                                                                                 
    {                                                                                                                                            
        printf("TEST PASSED ✓\n");                                                                                                                  
    }                                                                                                                                            
    else                                                                                                                                         
    {                                                                                                                                            
        printf("TEST FAILED ✗\n");                                                                                                                  
    }                                                                                                                                            
    
    // Clean up resources
    free(h_qkv);                                                                                                                                 
    free(h_kcache);                                                                                                                              
    free(h_vcache);                                                                                                                              
    free(h_o);                                                                                                                                   
    free(CPU_output);                                                                                                                            
    free(h_finished);                                                                                                                            
    free(h_qkv_bias);                                                                                                                            
    CHECK_CUDA_ERROR(cudaFree(d_finished));                                                                                                      
    CHECK_CUDA_ERROR(cudaFree(d_qkv));                                                                                                           
    CHECK_CUDA_ERROR(cudaFree(d_o));                                                                                                             
    CHECK_CUDA_ERROR(cudaFree(d_kcache));                                                                                                        
    CHECK_CUDA_ERROR(cudaFree(d_vcache));                                                                                                        
    CHECK_CUDA_ERROR(cudaFree(d_qkv_bias));
    
    delete qkv;
    delete kcache;
    delete vcache;
    delete finished;
    delete step;
    delete layer_id;
    delete mha_output;
    
    return is_true ? 0 : 1;
}