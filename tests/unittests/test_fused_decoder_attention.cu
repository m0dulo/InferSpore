#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>

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
                   const int head_size,
                   int step)
{
    int batch_stride = num_heads * head_size;
    int head_stride = head_size;
    int cache_offset = batch_size * batch_stride;
    int block_nums = batch_size * num_heads;
    float scale = 1 / sqrtf(float(head_size));

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
            float row_max = 0.0f;
            for (int iter = 0; iter < step; iter++)
            {
                float attn_score = 0.0f;
                for (int tid = 0; tid < head_size; tid++)
                {
                    int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                    sk[tid] = (float)k_cache[iter * cache_offset + qkv_offset];
                    if (iter == step - 1)
                    {
                        k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
                        sk[tid] = (float)k_mem[qkv_offset];
                    }

                    sq[tid] = (float)q_mem[qkv_offset];
                    float qk = sq[tid] * sk[tid] * scale;
                    attn_score += qk;
                }
                logits[iter] = attn_score;
                row_max = std::max(attn_score, row_max);
            }
            
            float fenmu = 0.0f;
            for (int iter = 0; iter < step; iter++)
            {
                float fenzi = expf(logits[iter] - row_max);
                fenmu += fenzi;
                logits[iter] = fenzi;
            }
            for (int iter = 0; iter < step; iter++)
            {
                logits[iter] = logits[iter] / fenmu;
            }
            
            for (int tid = 0; tid < head_size; tid++)
            {
                float O = 0.0f;
                int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                for (int iter = 0; iter < step; iter++)
                {
                    sv[tid] = (float)v_cache[iter * cache_offset + qkv_offset];
                    if (iter == step - 1)
                    {
                        v_cache[iter * cache_offset + qkv_offset] = v_mem[qkv_offset];
                        sv[tid] = (float)v_mem[qkv_offset];
                    }
                    O += sv[tid] * logits[iter];
                }
                mha_output[qkv_offset] = O;
            }
        }
    }

    free(sqk);
}

template <typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size)
{
    for (int i = 0; i < output_size; i++)
    {
        float GPUres = (float)GPUoutput[i];
        if (fabs(CPUoutput[i] - GPUres) > 1e-6)
        {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUres);
            return false;
        }
    }
    return true;
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
    cudaMalloc((void **)&d_qkv, sizeof(float) * qkv_size);                                                                                        
    float *h_kcache;                                                                                                                              
    float *d_kcache;                                                                                                                              
    int kcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_kcache = (float *)malloc(sizeof(float) * kcache_size);                                                                                      
    cudaMalloc((void **)&d_kcache, sizeof(float) * kcache_size);                                                                                  
    float *h_vcache;                                                                                                                              
    float *d_vcache;                                                                                                                              
    int vcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_vcache = (float *)malloc(sizeof(float) * vcache_size);                                                                                      
    cudaMalloc((void **)&d_vcache, sizeof(float) * vcache_size);                                                                                  
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
    cudaMalloc((void **)&d_o, sizeof(float) * o_size);                                                                                           
    bool *h_finished = (bool *)malloc(sizeof(bool) * batch_size);                                                                                
    bool *d_finished;                                                                                                                            
    cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size);                                                                                 
    for (int i = 0; i < batch_size; i++)                                                                                                         
    {                                                                                                                                            
        h_finished[i] = static_cast<bool>(0);                                                                                                    
    }                                                                                                                                            
    float *h_qkv_bias = (float *)malloc(sizeof(float) * (2 * kv_num_heads + num_heads) * head_size);                                             
    float *d_qkv_bias;                                                                                                                           
    cudaMalloc((void **)&d_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size);                                                
    for (int i = 0; i < (2 * kv_num_heads + num_heads) * head_size; i++)                                                                         
    {                                                                                                                                            
        h_qkv_bias[i] = (float)0.0f;                                                                                                             
    }                                                                                                                                            
    cudaMemcpy(d_qkv, h_qkv, sizeof(float) * batch_size * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);                   
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);                      
    cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice);                                                       
    cudaMemcpy(d_kcache, h_kcache, sizeof(float) * kcache_size, cudaMemcpyHostToDevice);                                                         
    cudaMemcpy(d_vcache, h_vcache, sizeof(float) * vcache_size, cudaMemcpyHostToDevice);                                                         
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
    launchDecoderMaskedMHA(qkv, qkv_weight, layer_id, kcache, vcache, finished, step, mha_output, params);                                       
    cudaMemcpy(h_o, d_o, sizeof(float) * o_size, cudaMemcpyDeviceToHost);                                                                 
    float *CPU_output = (float *)malloc(sizeof(float) * o_size);                                                                                 
    CPUMaskedAttn<float>(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, head_size, h_step);                               
    bool is_true = CheckResult<float>(CPU_output, h_o, o_size);                                                                                  
    if (is_true)                                                                                                                                 
    {                                                                                                                                            
        printf("test passed\n");                                                                                                                  
    }                                                                                                                                            
    else                                                                                                                                         
    {                                                                                                                                            
        printf("test failed\n");                                                                                                                  
    }                                                                                                                                            
    free(h_qkv);                                                                                                                                 
    free(h_kcache);                                                                                                                              
    free(h_vcache);                                                                                                                              
    free(h_o);                                                                                                                                   
    free(CPU_output);                                                                                                                            
    free(h_finished);                                                                                                                            
    cudaFree(d_finished);                                                                                                                        
    cudaFree(d_qkv);                                                                                                                             
    cudaFree(d_o);                                                                                                                               
    cudaFree(d_kcache);                                                                                                                          
    cudaFree(d_vcache);
}