#pragma once
#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/utils/weight_utils.h"
template<typename T>
class LlamaLayerWeight {
private:
    int     head_num;
    int     kv_head_num;
    int     head_size;
    int     hidden_units;
    int     inter_size;
    WeightType weight_type;
    int     bit_size;
    bool    attn_bias;

public:
    LlamaLayerWeight() = delete;
    LlamaLayerWeight(int head_num,
                    int  kv_head_num,
                    int  head_size,
                    int  inter_size,
                    WeightType weight_type,
                    bool attn_bias);
    ~LlamaLayerWeight();

    void loadWeights(std::string weight_path, WeightType weight_type);
    
    void loadWeights();

    LayerNormWeight<T> attn_norm_weight;
    LayerNormWeight<T> ffn_norm_weight;
    LLaMAattentionWeights<T> self_attn_weight;
    LLaMAFFNWeights<T> ffn_weight;
};