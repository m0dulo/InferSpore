#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <math.h>
#include "src/kernels/attn_softmax_kernel.h"
// (m0dulo)note:
// there is no cpu kernel implementation now, and if you bought my CUDA lesson, you can find CPU softmax kernel.
// we compare the kernel correctnesss by eyes and result print infos
// `./test_mask_softmax 1` to test half GPU kernel
// `./test_mask_softmax` to test fp32 GPU kernel
#define TEST_MASKED_SOFTMAX(dtype)                                                                                                  \
    dtype *h_qk;                                                                                                                    \
    dtype *d_qk;                                                                                                                    \
    h_qk = (dtype *)malloc(sizeof(dtype) * qk_size);                                                                                \
    cudaMalloc((void **)&d_qk, sizeof(dtype) * qk_size);                                                                            \
    dtype *h_score;                                                                                                                 \
    dtype *d_score;                                                                                                                 \
    h_score = (dtype *)malloc(sizeof(dtype) * qk_size);                                                                             \
    cudaMalloc((void **)&d_score, sizeof(dtype) * qk_size);                                                                         \
    dtype *h_mask;                                                                                                                \
    dtype *d_mask;                                                                                                                \
    h_mask = (dtype *)malloc(sizeof(dtype) * batch_size * q_length * k_length);                                                 \
    cudaMalloc((void **)&d_mask, sizeof(dtype) * batch_size * q_length * k_length);                                               \
    for (int i = 0; i < qk_size; i++)                                                                                               \
    {                                                                                                                               \
        h_qk[i] = i % 8;                                                                                                             \
    }                                                                                                                               \
    for (int i = 0; i < batch_size * q_length * k_length; i++)                                                                      \
    {                                                                                                                               \
        h_mask[i] = (dtype)(1);                                                                                                   \
    }                                                                                                                               \
    cudaMemcpy(d_qk, h_qk, sizeof(dtype) * qk_size, cudaMemcpyHostToDevice);                                                        \
    cudaMemcpy(d_mask, h_mask, sizeof(dtype) * batch_size * q_length * k_length, cudaMemcpyHostToDevice);                         \
    DataType type = getTensorType<dtype>();                                                                                         \
    TensorWrapper<dtype> *qk = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_qk);       \
    TensorWrapper<dtype> *mask = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, q_length, k_length}, d_mask);             \
    TensorWrapper<dtype> *score = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_score); \
    std::cout << "before launch softmax kernel" << std::endl;                                                                       \
    launchScaleMaskAndSoftmax(qk, mask, score, scale);                                                                              \
    std::cout << "after launch softmax kernel" << std::endl;                                                                        \
    std::cout << "cuda memcpy device to host" << std::endl;                                                                         \
    cudaMemcpy(h_score, score->data, sizeof(dtype) * qk_size, cudaMemcpyDeviceToHost);                                              \
    for (int i = 0; i < qk_size; i++)                                                                                               \
    {                                                                                                                               \
        printf("attn score[%d] = %f\n", i, (float)h_score[i]);                                                                      \
    }                                                                                                                               \
    free(h_qk);                                                                                                                     \
    free(h_score);                                                                                                                  \
    free(h_mask);                                                                                                                   \
    cudaFree(d_qk);                                                                                                                 \
    cudaFree(d_score);                                                                                                              \
    cudaFree(d_mask);

int main(int argc, char *argv[])
{
    const int batch_size = 1;
    const int head_num = 2;
    const int q_length = 8;
    const int k_length = 8;
    const int head_size = 4;
    float scale = rsqrtf(float(head_size));
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int qk_size = batch_size * head_num * q_length * k_length;
    if (argv[1])
    {
        TEST_MASKED_SOFTMAX(half);
    }
    else
    {
        TEST_MASKED_SOFTMAX(float);
    }
}
