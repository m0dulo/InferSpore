#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <math.h>
#include "src/kernels/repeat_kv.h"
// (m0dulo)note:
// there is no repeat kv cpu kernel implementation now
// we compare the kernel correctnesss by eyes
// `./test_repeat_kv` to test fp32 GPU kernel
int main() {
    const int batch_size = 1;
    const int head_num = 2;
    const int kv_head_num = 2;
    const int max_seq_len = 4;
    const int max_k_len = 2;
    const int head_size = 2;
    const int num_layers = 2;
    const int k_size = num_layers * batch_size * kv_head_num * max_seq_len * head_size;
    const int out_k_size = batch_size * head_num * max_k_len * head_size;
    float* h_k;
    float* d_k;
    h_k = (float*)malloc(sizeof(float) * k_size);
    cudaMalloc((void**)&d_k, sizeof(float) * k_size);
    float* h_v;
    float* d_v;
    h_v = (float*)malloc(sizeof(float) * k_size);
    cudaMalloc((void**)&d_v, sizeof(float) * k_size);
    int* h_ctx_len;
    int* d_ctx_len;
    h_ctx_len = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_ctx_len, sizeof(int) * batch_size);
    float* h_trans_k;
    float* d_trans_k;
    h_trans_k = (float*)malloc(sizeof(float) * out_k_size);
    cudaMalloc((void**)&d_trans_k, sizeof(float) * out_k_size);
    float* h_trans_v;
    float* d_trans_v;
    h_trans_v = (float*)malloc(sizeof(float) * out_k_size);
    cudaMalloc((void**)&d_trans_v, sizeof(float) * out_k_size);   

    for(int i = 0; i < k_size; i++) {
       h_v[i] = i;
       h_k[i] = i;
    }
    int* h_layer_id = (int*)malloc(sizeof(int)*batch_size);

    for(int i = 0; i < batch_size; i++) {
       h_ctx_len[i] = 2;
       h_layer_id[i] = 0;
    }    
    
    cudaMemcpy(d_k, h_k, sizeof(float) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(float) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    DataType type = getTensorType<float>(); 
    DataType type_int = getTensorType<int>(); 
    TensorWrapper<float>* in_k = new TensorWrapper<float>(Device::GPU, type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size}, d_k);
    TensorWrapper<float>* in_v = new TensorWrapper<float>(Device::GPU, type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size}, d_v);
    TensorWrapper<int>* ctx_len = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_ctx_len);
    TensorWrapper<float>* out_k = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size}, d_trans_k);
    TensorWrapper<float>* out_v = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size}, d_trans_v);
    TensorWrapper<int>* layer_id = new TensorWrapper<int>(Device::CPU, type_int, {batch_size}, h_layer_id);
    
    std::cout << "before launch repeat kv kernel" << std::endl;
    launchRepeatKVCache(in_k, in_v, ctx_len, layer_id, out_k, out_v);
    std::cout << "after launch repeat kv kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_trans_k, out_k->data, sizeof(float) * out_k_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < out_k_size; i++) {
        printf("k trans[%d] = %f\n", i, h_trans_k[i]);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_k);
    free(h_v);
    free(h_ctx_len);
    free(h_trans_k);
    free(h_trans_v);
    free(h_layer_id);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_ctx_len);
    cudaFree(d_trans_k);
    cudaFree(d_trans_v);
}
