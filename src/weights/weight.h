#pragma once
struct Weight {
    virtual void loadWeights(std::string weight_path) = 0;
};