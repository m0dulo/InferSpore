#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"

// 基础的块结构用于CUDA内存分配
// Basic block structure for CUDA memory allocation
struct CudaBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaBlock() = default;
    CudaBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};

/*
*   分配块的好处：在linux中需要进入内核态才能真正分配cudaMalloc，耗时很大，如果能维护块表
*   则可以大大减少分配的开销提高块的利用率
*/
class CudaAllocator: public BaseAllocator {
private:
    //{device id: block}
    std::map<int, std::vector<CudaBlock>> cudaBlocksMap;    
    size_t total_allocated_size = 0;  
    int dev_id;
public:
    CudaAllocator() {
        cudaGetDevice(&dev_id);
    }
    
    ~CudaAllocator() {
        for (auto &it: cudaBlocksMap) {
            auto &cudaBlocks = it.second; //vector
            for (int i = 0; i < cudaBlocks.size(); i++) {
                cudaFree(cudaBlocks[i].data);
            }
        }
    }

    void* UnifyMalloc(void* ptr, size_t size, bool is_host) {
        // 1. host malloc
        if (is_host) {
            //CHECK(cudaMallocHost(&ptr, size)); // for cuda stream async
            ptr = malloc(size);
            memset(ptr, 0, size);
            return ptr;
        }
        
        // 2. 在块池中查找可用的块
        auto &cudaBlocks = cudaBlocksMap[dev_id];
        int blockID = -1;
        for (int i = 0; i < cudaBlocks.size(); i++) {
            // 首次适应算法查找合适的块
            if (cudaBlocks[i].size >= size && !cudaBlocks[i].is_allocated) {
                if (blockID == -1 || cudaBlocks[blockID].size > cudaBlocks[i].size) {
                    blockID = i;
                }
            }
        }
        
        // 如果找到了合适的块
        if (blockID != -1) {
            cudaBlocks[blockID].is_allocated = true;
            std::cout << "已从现有块分配内存, id = " << blockID 
                    << ", 请求大小 = " << size << "B"
                    << ", 块大小 = " << cudaBlocks[blockID].size << "B"
                    << std::endl;
            return cudaBlocks[blockID].data;
        }
        
        // 否则分配一个新块
        void* new_buffer = (void*)ptr;
        CHECK(cudaMalloc(&new_buffer, size));
        CHECK(cudaMemset(new_buffer, 0, size));
        total_allocated_size += size;
        std::cout << "从CUDA分配新块, 大小 = " << size 
                << "B, 总分配量 = " << total_allocated_size << "B"
                << std::endl;
        
        cudaBlocks.push_back(CudaBlock(new_buffer, size, true));
        return new_buffer;
    }

    void UnifyFree(void* ptr, bool is_host) {
        if (ptr == nullptr) {
            return;
        }
        
        // 1. host free
        if (is_host) {
            free(ptr);
            return;
        }
        
        // 2. 找到并标记块为空闲（不实际释放给OS）
        for (auto &it: cudaBlocksMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {
                    cudaBlocks[i].is_allocated = false;
                    std::cout << "已释放块但保留在池中, 块id = " << i
                            << ", 大小 = " << cudaBlocks[i].size << "B"
                            << std::endl;
                    return;
                }
            }
        }
        
        // 如果在块池中未找到，直接释放
        std::cout << "块未在池中找到，直接释放" << std::endl;
        cudaFree(ptr);    
    }
};