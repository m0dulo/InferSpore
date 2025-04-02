#pragma once
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>
enum class WeightType
{
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};

template<typename T>
inline WeightType getWeightType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return WeightType::FP32_W;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return WeightType::FP16_W;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return WeightType::INT8_W;
    }
    else {
        return WeightType::UNSUPPORTED_W;
    }
}
template<typename T>
struct BaseWeight {
    std::vector<int> shape;
    T*   data;
    WeightType type;
    T*   bias;
};