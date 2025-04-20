#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template<typename T, int K>
struct topK
{
    T val[K];
    int id[K];

    __device__ void init(){
        for (int i = 0; i < K; i++) {
            id[i] = -1;
            val[i] = 0.0; 
        }
    }

    __device__ void insertHeap(T data, int data_id){
        if(id[K-1] == -1 || val[K-1] < data){
            id[K-1] = data_id;
            val[K-1] = data;
        }
    }
};


template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                             TensorWrapper<int> *topk_ids,
                             TensorWrapper<T> *topk_vals,
                             TensorWrapper<int> *final_topk_ids,
                             TensorWrapper<T> *final_topk_vals);