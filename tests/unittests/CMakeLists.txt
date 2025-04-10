add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(
  paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    cal_paddingoffset) 

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries( 
    -lcudart
    -lcudadevrt
    build_casual_mask) 

add_executable(biasRope
    test_bias_and_RoPE.cu
)
target_link_libraries(  
  biasRope PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope) 