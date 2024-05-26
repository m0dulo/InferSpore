#include <stdio.h>
#include "src/kernels/input_embedding.h"
#include "src/utils/cuda_debug_utils.cuh"
template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
               T* output, 
               const T* embed_table,
               const int max_context_token_num,
               const int hidden_size)
{
    // zhaziqwe：得到全局线程id
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // 如果分配的总线程数不够，那么在输出范围内的线程id再来多轮
    while (index < max_context_token_num * hidden_size) {
        // 拿到token_num这一个维度的序号，输出的行号
        int id = input_ids[index / hidden_size];
        // 通过token id索引到对应embedding table的行号，此行号乘上embedding的列数，得到该token id的hidden units
        // 每个线程并行地把hidden size个值读取并写到对应线程id位置的output
        output[index] = embed_table[id * hidden_size + index % hidden_size];
        // 当前线程处理完一轮，累加index到下一轮，防止总线程数不够。
        index += blockDim.x * gridDim.x;
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [token num]
                          TensorWrapper<T>* output,       // FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table// FP32 [vocab_size, hidden_size]
                          ) {
    // zhaziqwe：分配线程块，核函数需要的维度信息 
    const int blockSize = 256;
    const int max_context_token_num = output->shape[0]; // token num
    const int hidden_size = output->shape[1];
    const int gridSize = 2048;
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");
    embeddingFunctor<T><<<gridSize, blockSize>>>(input_ids->data,
                                                 output->data,
                                                 embed_table->data,
                                                 max_context_token_num,
                                                 hidden_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

// zhaziqwe: 显式实例化模版函数，由于cuda的语法规则，不能存在.cpp文件里，因此只能在此实例化
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table);
