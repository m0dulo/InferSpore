add_library(embeddingFunctor STATIC input_embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cal_paddingoffset STATIC cal_paddingoffset.cu)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cal_paddingoffset PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(build_casual_mask STATIC build_casual_mask.cu)
set_property(TARGET build_casual_mask PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET build_casual_mask PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET build_casual_mask PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cublasWrapper STATIC cublas_utils.cc)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(linear STATIC linear.cu)
set_property(TARGET linear PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET linear PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET linear PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(linear PUBLIC -lcudart -lcublas cublasWrapper)

add_library(qkv_bias_and_rope STATIC qkv_bias_and_RoPE.cu)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET qkv_bias_and_rope PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(concat_kv STATIC concat_past_kv.cu)
set_property(TARGET concat_kv PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET concat_kv PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET concat_kv PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(repeat_kv STATIC repeat_kv.cu)
set_property(TARGET repeat_kv PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET repeat_kv PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET repeat_kv PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)