#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include "src/utils/macro.h"
//1.cublas API: must allocate the required matrices in the GPU memory space, 
// fill them with data, call the sequence of desired cuBLAS functions, and then upload the results back to the host.
//2.cublasXt API: have the data on the Host
//3.cuBLASLt API: lightweight library dedicated to GEMM  with a new flexible API. 
// adds flexibility in matrix data layouts, input types, compute types, and also in choosing the algorithmic implementations and heuristics through parameter programmability
class cublasWrapper {
    private:
        cublasHandle_t   cublas_handle_;
        cublasLtHandle_t cublaslt_handle_;     

        cudaDataType_t Atype_;
        cudaDataType_t Btype_;
        cudaDataType_t Ctype_;
        cudaDataType_t computeType_;   
    
    public:
        cublasWrapper(cublasHandle_t cublas_handle_,
                      cublasLtHandle_t cublaslt_handle_);
                      // BaseAllocator* allocator); enable it when we use cublasLt API

        ~cublasWrapper();
        void setFP32GemmConfig();
        void setFP16GemmConfig();
        //for proj matmul
        void Gemm(cublasOperation_t transa,
                cublasOperation_t transb,
                const int         m,
                const int         n,
                const int         k,
                const void*       A,
                const int         lda,
                const void*       B,
                const int         ldb,
                void*             C,
                const int         ldc,
                float             alpha,
                float             beta);
        // for qk*v and q*k
        void stridedBatchedGemm(cublasOperation_t transa,
                                cublasOperation_t transb,
                                const int         m,
                                const int         n,
                                const int         k,
                                const void*       A,
                                const int         lda,
                                const int64_t     strideA,
                                const void*       B,
                                const int         ldb,
                                const int64_t     strideB,
                                void*             C,
                                const int         ldc,
                                const int64_t     strideC,
                                const int         batchCount,
                                float             f_alpha,
                                float             f_beta);
};
