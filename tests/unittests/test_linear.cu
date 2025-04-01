#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include <fstream>
#include "src/utils/macro.h"
#include "src/kernels/linear.h"
#include "src/weights/base_weights.h"

void CPUlinear(float* input, float* weight, float* output,
                int m, int k, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                output[i * n + j] += input[i * k + l] * weight[l * n + j];
            }
        }
    }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    for(int i = 0; i < output_size; i++) {
    if (i < 5) {
        printf("0th res, CPUoutput = %f, GPUoutput = %f\n", CPUoutput[i], GPUoutput[i]);
    }
    if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }

    }
    return true;
}

int main(int argc, char *argv[]) {
    const int seqlen = 13;
    const int hidden_units = 4096;
    const int vocab_size = 32;
    const int inter_size = 10;
    int hidden_units_2 = 0;
    int output_size = 0;

    hidden_units_2 = hidden_units * hidden_units;
    output_size = seqlen * hidden_units;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_w;
    float* d_w;
    h_w = (float*)malloc(sizeof(float) * hidden_units_2);
    cudaMalloc((void**)&d_w, sizeof(float) * hidden_units_2);
    for(int i = 0; i < hidden_units_2; i++) {
       h_w[i] = (float)(i % 3); // 1 2 1 2
    }

    float* h_in = (float*) malloc(sizeof(float) * hidden_units * seqlen);
    float* d_in;
    cudaMalloc((void**)&d_in, sizeof(float) * seqlen *  hidden_units);
    for(int i = 0; i < hidden_units * seqlen; i++) {
       h_in[i] = (float)(i % 3);
    }

    float* h_out = (float*) malloc(sizeof(float) * output_size);
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * output_size);
    CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * hidden_units * seqlen, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, h_w, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice));
    DataType type = getTensorType<float>();
    WeightType wtype = getWeightType<float>();
    TensorWrapper<float>* in = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_in);
    BaseWeight<float> weight;
    weight.shape = {hidden_units, hidden_units};
    weight.data = d_w;
    weight.type = wtype;
    TensorWrapper<float>* out;
    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_out);
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    cublas_wrapper->setFP32GemmConfig();
    // debug info, better to retain:
    std::cout << "before launch kernel" << std::endl;
    launchLinearGemm(in, weight, out, cublas_wrapper);
    // debug info, better to retain:
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain:
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
    float* CPUout = (float*) malloc(sizeof(float) * output_size);
    CPUlinear(h_in, h_w, CPUout, seqlen, hidden_units, hidden_units);

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