#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/base_weights.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,
                            BaseWeight<T>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<T>* mha_output,
                            LLaMAAttentionStaticParams& static_params);