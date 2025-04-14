#include "src/kernels/fused_transpose_and_remv_pad.h"
#include <iostream>
// [b,h,s,d]=>[b,s,h,d]=>[num tokens,h,d]
// padding_offset.shape = [num_tokens]
// `./test_fused_trans_remv_pad` to test fp32 kernel
int main() {
    const int batch_size = 2;
    const int head_num = 2;
    const int max_seq_len = 4;
    const int head_size = 2;
    const int num_tokens = 5;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int in_size = batch_size * head_num * max_seq_len * head_size;
    const int out_size = num_tokens * head_num * head_size;
    float* h_in;
    float* d_in;
    h_in = (float*)malloc(sizeof(float) * in_size);
    cudaMalloc((void**)&d_in, sizeof(float) * in_size);
    float* h_out;
    float* d_out;
    h_out = (float*)malloc(sizeof(float) * out_size);
    cudaMalloc((void**)&d_out, sizeof(float) * out_size);
    int* h_padding_offset;
    int* d_padding_offset;
    h_padding_offset = (int*)malloc(sizeof(int) * num_tokens);
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * num_tokens);

    //1st seqlen: 2, due to 1st seq, so its padding offset are all 0
    //2nd seqlen: 3, so its padding offset are all 4-2=2
    for(int i = 0; i < in_size; i++) {
       h_in[i] = i;
    }
    for(int i = 0; i < 2; i++) {
       h_padding_offset[i] = 0;
    } 
    h_padding_offset[2] = 2;  
    h_padding_offset[3] = 2;
    h_padding_offset[4] = 2;

    cudaMemcpy(d_in, h_in, sizeof(float) * in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * num_tokens, cudaMemcpyHostToDevice);

    DataType type = getTensorType<float>(); 
    DataType type_pad = getTensorType<int>(); 
    TensorWrapper<float>* in = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, max_seq_len, head_size}, d_in);
    TensorWrapper<int>* in_pad = new TensorWrapper<int>(Device::GPU, type_pad, {num_tokens}, d_padding_offset);
    TensorWrapper<float>* out = new TensorWrapper<float>(Device::GPU, type, {num_tokens, head_num, head_size}, d_out);
    std::cout << "before launch softmax kernel" << std::endl;
    launchTransposeOutRemovePadding(in, in_pad, out);
    std::cout << "after launch softmax kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_out, out->data, sizeof(float) * out_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < out_size; i++) {
        printf("after trans and remv pad, out[%d] = %f\n", i, h_out[i]);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_in);
    free(h_out);
    free(h_padding_offset);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_padding_offset);
}   