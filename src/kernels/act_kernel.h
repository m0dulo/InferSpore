#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out);