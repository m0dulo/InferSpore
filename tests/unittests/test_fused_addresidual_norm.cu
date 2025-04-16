#include <algorithm>  
#include <iostream>   
#include <math.h>      
#include <stdlib.h>    
#include <string>      
#include <vector>      

#include <iostream>
#include "src/kernels/fused_addresidual_norm.h"

#include <stdio.h>
// (m0dulo)note:
// `./test_fused_addresidual_norm` to test fp32 GPU kernel
// (m0dulo)note: this kernel's CPU implementation is absolutely right.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

void CPUfusedresidandRMSNorm(float* h_residual, float* h_decoder_out, float* h_bias, 
                                    float* h_scale, float eps, int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float input = 0.0f;
        for (int i = 0; i < hidden_units; i++) {
            input = h_decoder_out[b * hidden_units + i] +
                    h_residual[b * hidden_units + i] + h_bias[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < hidden_units; i++) {
            sum += input * input;
        }
        
        mean = (float)(sum / hidden_units);
        inv_fenmu = rsqrt(mean + eps);
        
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
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

int main() {
    const int num_tokens = 2;
    const int hidden_units = 32;
    const int total_size = num_tokens * hidden_units;
    float eps = 0.5f;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_residual;
    float* d_residual;
    h_residual = (float*)malloc(sizeof(float) * total_size);
    cudaMalloc((void**)&d_residual, sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++) { 
       h_residual[i] = 0.0f;
    }

    float* h_decoder_out = (float*) malloc(sizeof(float) * total_size);
    float* decoder_out = (float*) malloc(sizeof(float) * total_size);
    float* d_decoder_out;
    cudaMalloc((void**)&d_decoder_out, sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++) { 
       h_decoder_out[i] = 1.0f;
    }
    //bias
    float* h_bias = (float*) malloc(sizeof(float) * hidden_units);
    float* d_bias;
    cudaMalloc((void**)&d_bias, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
       h_bias[i] = 0.0f;
    }
    //rmsnorm weights
    float* h_scale = (float*) malloc(sizeof(float) * hidden_units);
    float* d_scale;
    cudaMalloc((void**)&d_scale, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
       h_scale[i] = 1.0f;
    }

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_bias, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, h_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<float>* decoder_out_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                        type_float,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_decoder_out);
    TensorWrapper<float>* residual_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                        type_float,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_residual);                                                                        
    BaseWeight<float> norm;
//    norm.bias = d_bias;
    LayerNormWeight<float> scale;
    scale.gamma = d_scale;
    // debug info, better to retain: 
    std::cout << "before launch kernel" << std::endl;
    launchFusedAddBiasResidualRMSNorm(residual_tensor, 
                                    decoder_out_tensor, 
                                    norm,
                                    d_scale,
                                    eps);
    // debug info, better to retain: 
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain: 
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));
    float* CPUout = (float*) malloc(sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++){
        CPUout[i] = 1.0f;
    }
    CPUfusedresidandRMSNorm(h_residual, CPUout, h_bias, 
                h_scale, eps, hidden_units, num_tokens);
    bool is_right = CheckResult(CPUout, decoder_out, total_size);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "fused addres and rmsnorm passed" << std::endl;
    free(h_residual);
    free(h_decoder_out);
    free(h_bias);
    free(h_scale);
    free(CPUout);
    free(decoder_out);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);
    cudaFree(d_bias);
    cudaFree(d_scale);
}