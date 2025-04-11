#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T>
void launchRepeatKVCache(TensorWrapper<T> *k_cache_src,
                         TensorWrapper<T> *v_cache_src,
                         TensorWrapper<int> *context_length,
                         TensorWrapper<int> *layer_id,
                         TensorWrapper<T> *k_cache_dst,
                         TensorWrapper<T> *v_cache_dst);