#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "src/models/basemodel.h"
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/models/llama/llama_params.h"
// (m0dulo) note: all LLM models are created in the header file, and I provided two ways, one is real weight model, the other is dummy weight model for functionality
namespace llm {
    template<typename T>
    BaseModel *CreateModelWithName(const std::string& model_name) {
        LLM_CHECK_WITH_INFO(model_name == "llama", "dont support other models except llama yet!");
        int head_num = 32;
        int kv_head_num = 32;
        int head_size = 128;
        int inter_size = 11008;
        int num_layers = 32;
        int max_seq_len = 64;
        int vocab_size = 32000;
        int hidden_units = (head_num + 2 * kv_head_num) * head_size;
        int q_hidden_units = head_num * head_size;
        bool attn_bias = false;
        LLaMAAttentionStaticParams attn_static_params;
        attn_static_params.rotary_embedding_dim = 128;
        attn_static_params.rotary_embedding_base = 10000;
        attn_static_params.max_position_embeddings = 4096;
        attn_static_params.use_dynamic_ntk = false; // true is for dyn scaling rope
        cublasHandle_t cublas_handle;
        cublasLtHandle_t cublaslt_handle;
        cudaStream_t stream;
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
        cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
        cublas_wrapper->setFP32GemmConfig();
	BaseAllocator* allocator = new CudaAllocator;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        BaseModel *model = new Llama<T>(head_num,
                                        kv_head_num,
                                        head_size,
                                        inter_size,
                                        num_layers,
                                        vocab_size,
                                        attn_static_params,
                                        max_seq_len,
                                        stream,
                                        cublas_wrapper,
                                        allocator,
                                        &deviceProp);
        return model;
    }

    template<typename T>
    std::unique_ptr<BaseModel> CreateDummyLLMModel(std::string tokenizer_file){
        BaseModel *model = CreateModelWithName<T>("llama");
        model->loadTokenizer(tokenizer_file);
        model->loadWeightsFromDummy();
        return std::unique_ptr<BaseModel> (model);        
    }

    template<typename T>
    std::unique_ptr<BaseModel> CreateRealLLMModel(std::string model_dir, std::string tokenizer_file){
        BaseModel *model = CreateModelWithName<T>("llama");
	std::cout << "start creating model..." << "\n";
	model->loadTokenizer(tokenizer_file);
        model->loadWeights(model_dir);
	std::cout << "finish creating model..." << "\n";
        return std::unique_ptr<BaseModel> (model);        
    }
} // namespace llm