#pragma once
#include <string>
#include <functional>
#include "src/utils/tensor.h"
#include "src/models/common_params.h"
#include "src/memory/allocator/base_allocator.h"
#include "src/kernels/cublas_utils.h"
// (m0dulo)note: 回调函数, 用于打印当前轮次对话的LLM生成内容
using CallBack = std::function<void(int index, const char* GenerateContent)>;

class BaseModel{
public:
    std::string model_name;
    // (m0dulo)note: 必需的且所有模型子类都共有的4个数据成员
    cudaStream_t stream;
    cublasWrapper* cublas_wrapper;
    BaseAllocator* allocator;
    cudaDeviceProp* cuda_device_prop;
    BaseModel(cudaStream_t stream,
              cublasWrapper* cublas_wrapper,
              BaseAllocator* allocator,
              cudaDeviceProp* cuda_device_prop = nullptr):
        stream(stream),
        cublas_wrapper(cublas_wrapper),
        allocator(allocator),
        cuda_device_prop(cuda_device_prop){};
    // (m0dulo)note: 3个纯虚函数API, 每个具体模型子类需要实现
    virtual void loadTokenizer(std::string file) = 0;
    virtual void loadWeights(std::string file) = 0;
    virtual void loadWeightsFromDummy() = 0;
    // (m0dulo)note: 3个纯虚函数API, 用于定义每轮对话的输入、历史记录和回复API
    // 根据历史信息和当前输入生成当前轮次的prompt
    virtual std::vector<std::string> MakeInput(const std::string &history, int round, const std::string &input) = 0;
    // 根据当前轮次回复更新到history string
    virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0;
    // 回复内容的返回接口
    virtual std::string Response(const std::vector<std::string>& input, CallBack PrintRes) = 0;
};