#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template<typename T>
void launchTransposeOutRemovePadding(TensorWrapper<T>* qkv_buf_w_pad, 
                                     TensorWrapper<int>* padding_offset,
                                     TensorWrapper<T>* qkv_buf_wo_pad_1);