#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"

// I use Bytes to printf buffer size msg, because sometime I allocate <1KB buffer, which causes that display 0KB
// 分了两种块 
struct CudaBigBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaBigBlock() = default;
    CudaBigBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};

struct CudaSmallBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaSmallBlock() = default;
    CudaSmallBlock(void* data_, int size_, bool is_allocated_):
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
    std::map<int, std::vector<CudaSmallBlock> > cudaSmallBlocksMap;    
    std::map<int, std::vector<CudaBigBlock> > cudaBigBlocksMap;
    std::map<int, size_t> FreeSize;  
    size_t total_allocated_size = 0;  
    int dev_id;
public:
    CudaAllocator() {
        cudaGetDevice(&dev_id);
    }
    ~CudaAllocator() {
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second; //vector
            for (int i = 0; i < cudaBlocks.size(); i++) {
                cudaFree(cudaBlocks[i].data);
            }
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                cudaFree(bigBlocks[i].data);
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
        // 2.big buf, 先去bigblocks里面找空闲的（free出来且未归还到OS的）
        if (size > 1024 * 1024) { // > 1M
            auto &BigBlocks = cudaBigBlocksMap[dev_id];
            int blockID = -1;
            for (int i = 0; i < BigBlocks.size(); i++) {
                // the freed bigblock by free method
                // 朴素的分配算法 分配策略为首次分配并且块空间-分配空间大于1M
                // 会造成碎片化
                if (BigBlocks[i].size >= size && !BigBlocks[i].is_allocated
                    && BigBlocks[i].size - size < 1 * 1024 * 1024) {
                    if (blockID == -1 || BigBlocks[blockID].size > BigBlocks[i].size) {
                        blockID = i;
                    }
                }
            }
            // the allocated big block id
            if (blockID != -1) {
                BigBlocks[blockID].is_allocated = true;
                // std::cout << "allocate a existed big block, id = " << blockID 
                //                                 <<", size = "<< size << "B"
                //                                 <<", block size = "<< BigBlocks[blockID].size << "B"
                //                                 << std::endl;
                                                
                return BigBlocks[blockID].data;
            }
            // 没找到空闲的再cudaMalloc，并插进block pool
            void* new_buffer;
            cudaMalloc(&new_buffer, size);
            total_allocated_size += size;
            // std::cout << "allocate a new big block from OS using cudaMalloc, size = "
            //                                     << size << "B, total allocated size " << total_allocated_size << "B"
            //                                     << std::endl;
            BigBlocks.push_back(CudaBigBlock(new_buffer, size, true));
            return new_buffer;
        }
        // 3.small buf, 先去smallblocks里面找空闲的（free出来且未归还到OS的）
        // 问题: 为什么要分成大小block? 答: 用free size记录碎片
        // 匹配策略：简单首次匹配
        auto &SmallBlocks = cudaSmallBlocksMap[dev_id];
        for (int i = 0; i < SmallBlocks.size(); i++) {
            if (SmallBlocks[i].size >= size && !SmallBlocks[i].is_allocated) {
                SmallBlocks[i].is_allocated = true;
                FreeSize[i] += SmallBlocks[i].size;//小buf size
                // std::cout << "allocate a existed small block, id = " << i 
                //                 <<", size = "<< size << "B"
                //                 <<", block size = "<< SmallBlocks[i].size << "B"
                //                 << std::endl;
                return SmallBlocks[i].data;
            }
        }
        // 4.没找到空闲的再cudaMalloc
        void* new_buffer = (void*)ptr;
        CHECK(cudaMalloc(&new_buffer, size));
        CHECK(cudaMemset(new_buffer, 0, size));
        // std::cout << "allocate a new small block from OS using cudaMalloc, size = "
        //                                     << size  << "B, total allocated size " << total_allocated_size << "B"
        //                                     << std::endl;

        SmallBlocks.push_back(CudaSmallBlock(new_buffer, size, true));
        return new_buffer;
    }

    void UnifyFree(void* ptr, bool is_host) {
        if (ptr == nullptr) {
            return;
        }
        // 1.host free
        if (is_host) {
            free(ptr);
            return;
        }
        // 2.清理碎片：当累计的小buf超出了1G时，清理未分配出去的smallblocks, 已分配的还是保留在smallmap
        for (auto &it: cudaSmallBlocksMap) {
            if (FreeSize[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> temp;
                for (int i = 0; i < cudaBlocks.size(); i++) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        // std::cout << "free a small block to OS using cudaFree, block id = "
                        //                                     << i
                        //                                     << ",size = "
                        //                                     << cudaBlocks[i].size << "B"
                        //                                     << std::endl;
                        cudaFree(cudaBlocks[i].data);
                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = temp;
                FreeSize[it.first] = 0;
            }
        }
        // 3.找到待free的buffer的位置，设is_allocated = false，大小block都不归还到OS，除非没有在大小block里面找到待free的ptr
        // 大块清理分配比较耗时，为了降低损耗，用标记为标记为已经清除即可
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {
                    FreeSize[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
                    // std::cout << "free a small block but not to OS, block id = "
                    //                                     << i
                    //                                     << ",size = "
                    //                                     << cudaBlocks[i].size << "B"
                    //                                     << std::endl;
                    return;
                }
            }
            //若是大block，那不归还到OS
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    // std::cout << "free a big block but not to OS, block id = "
                    //                                     << i
                    //                                     << ",size = "
                    //                                     << cudaBlocks[i].size << "B"
                    //                                     << std::endl;
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }
        // std::cout << "NOT found the ptr in blocks, so free the ptr to OS using cudaFree"
                                            // << std::endl;
        cudaFree(ptr);    
    }
};