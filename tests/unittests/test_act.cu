#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "src/kernels/act_kernel.h"
#include "src/utils/tensor.h"

template<typename T>
void CPUSwiGLU(T* input, T* output, int batch_size, int intermedia_size){
    float silu_out = 0.0f;
    for(int batch_id = 0; batch_id < batch_size; batch_id++){
        for(int i = 0; i < intermedia_size; i++) {
            int offset1 = batch_id * 2 * intermedia_size + i;
            int offset2 = batch_id * 2 * intermedia_size + i + intermedia_size;
            int out_offset = batch_id * intermedia_size + i;
            silu_out = (float)input[offset1] / (1.0f + expf(-1.0f * (float)input[offset1]));
            output[out_offset] = static_cast<T>(silu_out * (float)input[offset2]);
        }
    }
}

template<typename T>
bool CheckResult(T* CPUoutput, T* GPUoutput, int output_size) {
    bool correct = true;
  
    float tolerance = std::is_same<T, half>::value ? 1e-3 : 1e-5;
    for(int i = 0; i < output_size; i++) {
        float cpu_val = (float)CPUoutput[i];
        float gpu_val = (float)GPUoutput[i];
        if(fabs(cpu_val - gpu_val) > tolerance){
            printf("the %dth res is wrong (tolerance %f), CPUoutput = %f, GPUoutput = %f\n", i, tolerance, cpu_val, gpu_val);
            correct = false;
        }
    }
    return correct;
}

template<typename T> DataType getTensorType();
template<> DataType getTensorType<float>() { return DataType::FP32; }
template<> DataType getTensorType<half>() { return DataType::FP16; }


template<typename T>
void test_act(int batch_size, int intermedia_size, int input_size , int output_size) {
    T* h_input;
    T* d_input;
    h_input = (T*)malloc(sizeof(T) * input_size);
    cudaMalloc((void**)&d_input, sizeof(T) * input_size);
    T* h_output;
    T* d_output;
    h_output = (T*)malloc(sizeof(T) * output_size);
    cudaMalloc((void**)&d_output, sizeof(T) * output_size);
    
    for(int i = 0; i < input_size; i++) {
         h_input[i] = static_cast<T>( (rand() % 100) / 50.0f - 1.0f ); // Random float between -1.0 and 1.0
    }

    cudaMemcpy(d_input, h_input, sizeof(T) * input_size, cudaMemcpyHostToDevice);
    DataType type = getTensorType<T>();
   
    TensorWrapper<T>* input_tensor = new TensorWrapper<T>(Device::GPU, type, {batch_size, 2, intermedia_size}, d_input);
    TensorWrapper<T>* output_tensor = new TensorWrapper<T>(Device::GPU, type, {batch_size, intermedia_size}, d_output);
    launchAct(input_tensor, output_tensor);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, sizeof(T) * output_size, cudaMemcpyDeviceToHost);
    T* CPU_output = (T*)malloc(sizeof(T) * output_size);
    CPUSwiGLU(h_input, CPU_output, batch_size, intermedia_size);
    bool is_true = CheckResult(CPU_output, h_output, output_size);
     if(is_true){
        printf("test passed for type %s\n", (type == DataType::FP32 ? "FP32" : "FP16"));
    } else {
        printf("test failed for type %s\n", (type == DataType::FP32 ? "FP32" : "FP16"));
    }

    free(h_input);
    free(h_output);
    free(CPU_output);
    cudaFree(d_input);
    cudaFree(d_output);
    delete input_tensor;
    delete output_tensor;
}

int main(int argc, char** argv) {
    srand(1234); // Seed for reproducible random numbers
    constexpr int batch_size = 16;
    constexpr int intermedia_size = 11008;
    constexpr int input_size = batch_size * intermedia_size * 2;
    constexpr int output_size = batch_size * intermedia_size;

    if (argc > 1 && std::string(argv[1]) == "1") {
         printf("Testing FP16...\n");
         test_act<half>(batch_size, intermedia_size, input_size, output_size);
    } else {
         printf("Testing FP32...\n");
         test_act<float>(batch_size, intermedia_size, input_size, output_size);
    }
    return 0;
}