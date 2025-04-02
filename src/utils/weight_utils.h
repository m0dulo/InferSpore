#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include "src/utils/macro.h"

template<typename T>
void GPUMalloc(T** ptr, size_t size);

template<typename T>
void GPUFree(T* ptr);

template <typename T_OUT, typename T_FILE, bool Enabled = std::is_same<T_OUT, T_FILE>::value> struct loadWeightFromBin{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename);
}; 
