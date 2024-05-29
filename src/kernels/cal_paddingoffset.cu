#include "src/kernels/cal_paddingoffset.h"
// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
// TODO: we first use serial algo, then can enhance to CUDA scan algo

// bs = 3, seqlen = [3,2,5], max_seq_len = 5
// 1 1 1 0 0
// 1 1 0 0 0
// 1 1 1 1 1
// cum_seqlens:[0,3,2,5]
// paddingoffset 为
// 0 0 0 0 0
// 2 2 2 2 2
// 5 5 5 5 5
__global__ void CalPaddingoffset(int*         padding_offset, 
                                 int*         cum_seqlens,
                                 const int*   input_lengths, //actual input lens
                                 const int    batch_size,
                                 const int    max_q_len) {
    int ind = 0;
    int cum_offset = 0;
    int total_seqlen = 0;
    // 遍历每一个批次
    for(int b = 0; b < batch_size; b++) {
        // 获取到每个句子的长度
        int seqlen = input_lengths[b];
        // 累计的句子长度
        cum_seqlens[b] = total_seqlen;
        // 遍历一个句子里的所有token位置
        // each token in one seq has same cum offset
        for (int i = 0; i < seqlen; i++) {
            // index是对于每个token的索引，每个token都有一个paddingoffset
            padding_offset[ind] = cum_offset;
            ind++;
        }
        // 获取累计的 padding offset 和 总共的句子长度
        cum_offset += max_q_len - seqlen;
        total_seqlen += seqlen;
    }
    // 注意 cum_seqlens 的形状，添加最后一个累计句子长度（总长度）
    cum_seqlens[batch_size] = total_seqlen;
}

// 在attention之后，可以方便的移除padding。
// padding操作和 seq len 维度相关，因此相关操作需要在不涉及这一维度的计算后添加。  
void launchCalPaddingoffset(TensorWrapper<int>* padding_offset, 
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths)//actual input lens
{
    const int batch_size = padding_offset->shape[0];                            
    const int max_q_len = padding_offset->shape[1]; 
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0], "input lenghts numbers should equal to padding offset bs dim!") ;                        
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0] - 1, "cum seqlen numbers should equal to padding offset bs dim + 1!") ;                        
    CalPaddingoffset<<<1, 1>>>( 
        padding_offset->data, cum_seqlens->data, input_lengths->data, batch_size, max_q_len
    );
}