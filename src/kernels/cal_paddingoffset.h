#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/macro.h"
#include "src/utils/tensor.h"

void launchCalPaddingoffset(TensorWrapper<int>* padding_offset, 
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths //actual input lens
);