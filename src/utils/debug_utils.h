#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/macro.h"
// (m0dulo)note: overloaded 3 different function for saving intermediate output tensor to debug
// because LLMs have many layers, so I provide some overloaded function to specify layer id to print specify layer output tensor to debug
// after you save tensor into specified file ,you can turn to tests/unitests/test_data_compare.cu to specify file path to compare res with HF.
template<typename T>
void save_tensor(TensorWrapper<T>* input, std::string filename){
    int Bm = 0;
    int Bk = 0;
    if (input->shape.size() == 4){
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3){
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 2){
        Bm = input->shape[0];
        Bk = input->shape[1];
    }
    T* icpu = (T*)malloc(sizeof(T) * Bm * Bk);
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);
    std::ofstream F;
    std::cout << "saving intermediate tensor in " << filename << "\n";
    F.open("/home/data/"+ filename, std::ofstream::binary);
    F.write(reinterpret_cast<const char*>(icpu), sizeof(T)*Bm*Bk);
    F.close();
}

template<typename T>
void save_tensor(TensorWrapper<T>* input, std::string filename, TensorWrapper<int>* layer_id){
    int id = layer_id->getVal();
    if (id > 2) {
        return;
    }
    int Bm = 0;
    int Bk = 0;
    if (input->shape.size() == 4){
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3){
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 2){
        Bm = input->shape[0];
        Bk = input->shape[1];
    }
    T* icpu = (T*)malloc(sizeof(T) * Bm * Bk);
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);
    std::ofstream F;
    std::cout << "saving intermediate tensor in " << filename << "\n";
    F.open("/home/data/" + std::to_string(id) + "_" + filename, std::ofstream::binary);
    F.write(reinterpret_cast<const char*>(icpu), sizeof(T)*Bm*Bk);
    F.close();
}

template<typename T>
void save_tensor(TensorWrapper<T>* input, std::string filename, int layer_id){
    int id = layer_id;
    if (id > 2) {
        return;
    }
    int Bm = 0;
    int Bk = 0;
    if (input->shape.size() == 4){
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3){
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 2){
        Bm = input->shape[0];
        Bk = input->shape[1];
    }
    T* icpu = (T*)malloc(sizeof(T) * Bm * Bk);
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);
    std::ofstream F;
    std::cout << "saving intermediate tensor in " << filename << "\n";
    F.open("/home/data/" + std::to_string(id) + "_" + filename, std::ofstream::binary);
    F.write(reinterpret_cast<const char*>(icpu), sizeof(T)*Bm*Bk);
    F.close();
}
