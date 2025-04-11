
#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include "src/utils/macro.h"
#include "src/kernels/linear.h"
#include "src/weights/base_weights.h"
// (m0dulo)note: this kernel's CPU implementation is absolutely right.
// But when you are implementing LLMs inference on CPU, I dont recommend to reuse the CPU kernel, because its performance is bad
void CPUlinear(float* input, float* weight, float* output,
                int m, int k, int n, int batch) {
    for(int b = 0; b < batch; b++) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                for(int l = 0; l < k; l++) {
                    output[b * m * n + i * n + j] += input[b * m * k + i * k + l] * weight[b * k * n + l * n + j];
                }
            }
        }
    }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    for(int i = 0; i < output_size; i++) {
        if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }
    }
    return true;
}
// (m0dulo)note:
// `./bmm 1` to test fp32 GPU batch matmul with trans_b = true
// `./bmm` to test fp32 GPU batch matmul with trans_b = false
int main(int argc, char *argv[]) {
    const int batch_size = 1;
    const int seqlen_in = 16;
    const int seqlen_w = 16;
    const int hidden_units = 4096;
    const int head_num = 32;
    const int head_size = 128;
    int in_size = 0;
    int w_size = 0;
    int output_size = 0;
    if (argv[1]) {// enable trans_b for test lmhead linear
        in_size = batch_size * head_num * seqlen_in * head_size; // q
        w_size = batch_size * head_num * seqlen_w * head_size; // k
        output_size = batch_size * head_num * seqlen_in * seqlen_w; //q k
    } else {
        in_size = batch_size * head_num * seqlen_in * seqlen_w; //qk
        w_size = batch_size * head_num * seqlen_w * head_size; // v
        output_size = batch_size * head_num * seqlen_in * head_size;
    }
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_w;
    float* d_w;
    h_w = (float*)malloc(sizeof(float) * w_size);
    cudaMalloc((void**)&d_w, sizeof(float) * w_size);
    for(int i = 0; i < w_size; i++) { 
        h_w[i] = (float)(i % 2 + 1);
    	//h_w[i] = 1.0f; // simple data
    }

    float* h_in = (float*) malloc(sizeof(float) * in_size);
    float* d_in;
    cudaMalloc((void**)&d_in, sizeof(float) * in_size);
    for(int i = 0; i < in_size; i++) { 
        h_in[i] = (float)(i % 2 + 1);
    	//h_in[i] = 1.0f; // simple data
    }

    float* h_out = (float*) malloc(sizeof(float) * output_size);
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * output_size);

    CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * in_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, h_w, sizeof(float) * w_size, cudaMemcpyHostToDevice));
    DataType type = getTensorType<float>();
    WeightType wtype = getWeightType<float>(); 
    TensorWrapper<float>* in;
    if (argv[1]) {// enable trans_b for test qk*v
        in = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, seqlen_in, head_size}, d_in);
    } else {// disable trans_b for test q*k
        in = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, seqlen_in, seqlen_w}, d_in);
    }
    TensorWrapper<float>* weight = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, seqlen_w, head_size}, d_w);
    TensorWrapper<float>* out;
    if (argv[1]) {// enable trans_b for test qk*v
        out = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, seqlen_in, seqlen_w}, d_out);
    } else {// disable trans_b for test q*k
        out = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, seqlen_in, head_size}, d_out);
    }
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);  
    cublas_wrapper->setFP32GemmConfig();  
    // debug info, better to retain: 
    std::cout << "before launch kernel" << std::endl;
    if (argv[1]) {// enable trans_b for test qk*v
        launchLinearStridedBatchGemm(in, weight, out, cublas_wrapper, false, true);
    } else {// disable trans_b for test q*k
        launchLinearStridedBatchGemm(in, weight, out, cublas_wrapper);
    } 
    // debug info, better to retain: 
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain: 
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
    float* CPUout = (float*) malloc(sizeof(float) * output_size);
    if (argv[1]) {// enable trans_b for ttest qk*v
        CPUlinear(h_in, h_w, CPUout, seqlen_in, head_size, seqlen_w, batch_size * head_num);
    } else {// disable trans_b for test q*k
        CPUlinear(h_in, h_w, CPUout, seqlen_in, seqlen_w, head_size, batch_size * head_num);
    } 
    
    bool is_right = CheckResult(CPUout, h_out, output_size);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "linear passed" << std::endl;
    free(h_in);
    free(h_w);
    free(h_out);
    free(CPUout);
    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);
}