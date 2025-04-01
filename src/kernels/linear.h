#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include "src/kernels/cublas_utils.h"
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/macro.h"

template<typename T>
void launchLinearGemm(TensorWrapper<T>* input,
                      BaseWeight<T>& weight, 
                      TensorWrapper<T>* output,
                      cublasWrapper* cublas_wrapper,
                      bool trans_a = false,
                      bool trans_b = false);
template<typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T>* input1,
                                  TensorWrapper<T>* input2,
                                  TensorWrapper<T>* output,
                                  cublasWrapper* cublas_wrapper,
                                  bool trans_a = false,
                                  bool trans_b = false);
