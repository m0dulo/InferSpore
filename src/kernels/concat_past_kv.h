#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T>
void launchConcatKVCache(TensorWrapper<T> *k_src, 
                          TensorWrapper<T> *v_src,
                          TensorWrapper<int> *layer_id,         
                          TensorWrapper<int> *cur_query_length, 
                          TensorWrapper<int> *history_length,
                          TensorWrapper<T> *k_dst,
                          TensorWrapper<T> *v_dst); 