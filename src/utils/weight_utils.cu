#include "src/utils/weight_utils.h"

template<typename T_OUT, typename T_IN>
inline __device__ T_OUT type_cast(T_IN val) {
    return val;
}
template<>
inline __device__ float type_cast(half val) {
    return __half2float(val);
}

template<>
inline __device__ half type_cast(float val) {
    return __float2half(val); 
}

template<typename T>
void GPUMalloc(T** ptr, size_t size)
{
    LLM_CHECK_WITH_INFO(size >= ((size_t)0), "Ask cudaMalloc size " + std::to_string(size) + "< 0 is invalid.");
    CHECK(cudaMalloc((void**)(ptr), sizeof(T) * size));
}
template void GPUMalloc(float** ptr, size_t size);
template void GPUMalloc(half** ptr, size_t size);

template<typename T>
void GPUFree(T* ptr)
{
    if (ptr != NULL) {
        CHECK(cudaFree(ptr));
        ptr = NULL;
    }
}
template void GPUFree(float* ptr);
template void GPUFree(half* ptr);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size)
{
    CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, const float* src, const size_t size);
template void cudaH2Dcpy(half* tgt, const half* src, const size_t size);

template<typename T_IN, typename T_OUT>
__global__ void type_conversion(T_OUT* dst, const T_IN* src, const int size)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread_nums = blockDim.x * gridDim.x;
    for (int index = gtid; index < size; index += total_thread_nums) {
        dst[index] = type_cast<T_OUT>(src[index]);
    }
}

template<typename T_IN, typename T_OUT>
void cuda_type_conversion(T_OUT* dst, const T_IN* src, const int size)
{
    dim3 grid(128);
    dim3 block(128);
    type_conversion<T_IN, T_OUT><<<grid, block, 0, 0>>>(dst, src, size);
}

template void cuda_type_conversion(float* dst, const half* src, const int size);
template void cuda_type_conversion(half* dst, const float* src, const int size);


template<typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return std::vector<T>();
    }
    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        std::cout << "shape is zero, skip loading weight from file: " << filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "file" << filename << "cannot be opened, loading model fails!" << std::endl;
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    std::cout << "Read " << std::to_string(loaded_data_size) << " bytes from " << filename << std::endl;
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<T>();
    }
    in.close();
  
    return host_array;
}

template <typename T_OUT, typename T_FILE>
struct loadWeightFromBin<T_OUT, T_FILE, true>
{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename) {
        std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename);
        if (host_array.empty()) {
            return;
        }

        cudaH2Dcpy(ptr, host_array.data(), host_array.size());
        return;    
   }
};

template <typename T_OUT, typename T_FILE>
struct loadWeightFromBin<T_OUT, T_FILE, false>
{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename) {
        std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename);
        if (host_array.empty()) {
            return;
        }

        T_FILE* ptr_tmp;
        GPUMalloc(&ptr_tmp, host_array.size());
        cudaH2Dcpy(ptr_tmp, host_array.data(), host_array.size());
        cuda_type_conversion(ptr, ptr_tmp, host_array.size());
        GPUFree(ptr_tmp);
        return;
    }
};


template<typename T_OUT, typename T_FILE>
typename std::enable_if<std::is_same<T_OUT, T_FILE>::value, int>::type loadWeightFromBin(T_OUT* ptr, std::vector<size_t> shape, std::string filename)
{
    std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename);

    if (host_array.empty()) {
        return 0;
    }

    cudaH2Dcpy(ptr, host_array.data(), host_array.size());
    return 0;
}


template struct loadWeightFromBin<float, float, true>;
template struct loadWeightFromBin<half, half, true>;
template struct loadWeightFromBin<float, half, false>;
template struct loadWeightFromBin<half, float, false>;
