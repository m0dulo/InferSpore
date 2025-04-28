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

add_executable(test_concat_kv
    test_concat_kv.cu
)
target_link_libraries( 
    test_concat_kv PUBLIC
    -lcudart
    -lcudadevrt
    concat_kv) 

add_executable(testlinear
    test_linear.cu
)
target_link_libraries( 
    testlinear PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

add_executable(bmm
    test_bmm.cu
)
target_link_libraries( 
    bmm PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

add_executable(test_repeat_kv
    test_repeat_kv.cu
)
target_link_libraries( 
    test_repeat_kv PUBLIC
    -lcudart
    -lcudadevrt
    repeat_kv) 

add_executable(test_mask_softmax
    test_mask_softmax.cu
)
target_link_libraries(  
    test_mask_softmax PUBLIC
    -lcudart
    -lcudadevrt
    mask_softmax) 

add_executable(test_fused_trans_remv_pad
    test_fused_trans_remv_pad.cu
)
target_link_libraries(  
    test_fused_trans_remv_pad PUBLIC
    -lcudart
    -lcudadevrt
    fused_transpose_and_remv_pad) 

add_executable(test_act
    test_act.cu
)
target_link_libraries(  
    test_act PUBLIC
    -lcudart
    -lcudadevrt
    act) 

add_executable(test_topk
    test_topk.cu
)
target_link_libraries(  
    test_topk PUBLIC
    -lcudart
    -lcudadevrt
    topk) 

add_executable(test_fused_decoder_attention
    test_fused_decoder_attention.cu
)
target_link_libraries(  
    test_fused_decoder_attention PUBLIC
    -lcudart
    -lcudadevrt
    fused_decoder_self_attention)