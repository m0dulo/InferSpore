#pragma once
#include "src/weights/base_weights.h"
template<typename T>
struct LLaMAFFNWeights {
    BaseWeight<T> gate;
    BaseWeight<T> up;
    BaseWeight<T> down;
    BaseWeight<T> gateAndup;
};
