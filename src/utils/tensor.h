#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <cuda_fp16.h>
#include "src/utils/string_utils.h"
#include "src/utils/macro.h"
enum Device
{
    CPU_PINNED,
    CPU,
    GPU
};

enum DataType
{
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return FP32;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return FP16;
    }
    else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return INT32;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return INT8;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return BYTES;
    }
    else {
        return UNSUPPORTED;
    }
}
template<typename T>
class TensorWrapper;

struct Tensor {
    Device              location;
    DataType            dtype;
    std::vector<int>    shape;

    Tensor() = default;

    Tensor(const Device location_, 
            const DataType dtype_,
            const std::vector<int> shape_):
            location(location_),
            dtype(dtype_),
            shape(shape_){}
    virtual int size() const {
        if (shape.size() == 0) {
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }
// note: data's destructor is invoked by allocator's free API or cudaFree API            
//    ~Tensor() {
//        if(data!=nullptr) {
//            delete data;
//            data = nullptr;
//        }
//    }
    template<typename T>
    TensorWrapper<T>* as(){
        return static_cast<TensorWrapper<T>*>(this);
    }

    std::string DeviceString() const
    {
        static const std::unordered_map<Device, std::string> devicetring{
            {CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}};
        return devicetring.at(location);
    }

    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"},
            {INT32,"INT32"},
            {FP16, "FP16"},
            {FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str());
    }  
};

template<typename T>
class TensorWrapper: public Tensor {
public:
    T* data;
    // cant declare shape's type to std::vector<int>&, because we usually pass a tmp var, which cant be non-const refer
    TensorWrapper(Device location, DataType dtype, std::vector<int> shape):
    	Tensor(location, dtype, shape){}

    TensorWrapper(Device location, DataType dtype, std::vector<int> shape, T* data):
    	Tensor(location, dtype, shape),
    	data(data){
            DataType in_dtype = getTensorType<T>();
            LLM_CHECK_WITH_INFO(in_dtype == dtype, "when build TensorWrapper, the passed in data type should be same as dtype in params");
        }

    // friend bool operator==(Tensor& t1, Tensor& t2);
    virtual int size() const {
        if (data == nullptr || shape.size() == 0) {
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }

    inline T getVal(int id) const {
        //TODO: need some boundry and device check
        LLM_CHECK(location == CPU);
        return data[id];
    } // only available on CPU by []

    inline T getVal() const
    {
        // TODO: add type check, this is very important, because we often naturally access GPU data, which is wrong
        // for example, I am in transpose kernel to use layer_id->getVal<int>(), which is wrong
        LLM_CHECK(location == CPU);
        return getVal(0);
    }

    inline T* getPtr() const {
        //TODO: need some boundry check
        return (T*)data;
    }

    inline T* getPtrByOffset(int offset) const {
        //TODO: need some boundry check
        return (T*)data + offset;
    }
    // for debug
    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"},
            {FP16, "FP16"},
            {FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str(),
                    data);
    }    
};



//I cant check if the data pointer in TensorWrapper is nullptr, because the val in tensormap is tensor*
//so I must check the data pointer using LLM_CHECK_WITH_INFO before insert into tensormap.
struct TensorMap {
    std::unordered_map<std::string, Tensor*> tensor_map_;

    TensorMap() = default;
    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for (auto& pair : tensor_map) {
            if (isValid(pair.second)) {
                insert(pair.first, pair.second);
            }
            else {
                // std::cout << "this is not a valid tensor, skip to insert into tensormap" << std::endl;
                LLM_CHECK_WITH_INFO(isValid(pair.second),fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }

    TensorMap(const std::unordered_map<std::string, Tensor*>& tensor_map) {
        // C++ 11 traverse
        // for (auto& kv : tensor_map) {
        // C++ 98 traverse
        for(auto it = tensor_map_.begin(); it != tensor_map_.end(); it++) {
            // if (isValid(kv.second)) {
            //     insert(kv.first, kv.second);
            // }
            if (isValid(it->second)) {
                insert(it->first, it->second);
            }
            else {
                // TODO: add a reminder info
            }
        }        
    };

    ~TensorMap(){
        tensor_map_.clear();
    }

    inline size_t size() const
    {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const
    {
        return tensor_map_.find(key) != tensor_map_.end();
    }

    inline bool isValid(const Tensor* tensor)
    {
        return tensor->size() > 0;
    }
    // 增
    inline void insert(const std::string& key, Tensor* value)
    {
        // TODO: add a check to check key is unique and value is valid
        // tensor_map_.insert({key, value});
        tensor_map_[key] = value;
    }

    inline void insert(std::pair<std::string, Tensor*> p)
    {
        tensor_map_.insert(p);
    }
    //删

    //改

    //查
    inline Tensor* at(const std::string& key)
    {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
        
    }

    inline Tensor* operator[](const std::string& key)
    {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map    (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);

    }
    // TODO: Now cant use get* function in TensorMap struct, cause the value is Tensor*, not TensorWrapper,need to enhance
    // template<typename T>
    // inline T getVal(const std::string& key) const
    // {
    //     // TODO: add a check to check key is existed
    //     return tensor_map_.at(key).getVal<T>();
    // }
    // template<typename T>
    // inline T getValByOffset(const std::string& key, int index) const
    // {
    //     // TODO: add a check to check key is existed
    //     return tensor_map_.at(key).getVal<T>(index);
    // }
    // //default get ptr with offset 0
    // template<typename T>
    // inline T* getPtr(const std::string& key) const
    // {
    //     // TODO: add a check to check key is existed
    //     return tensor_map_.at(key).getPtr<T>();
    // }
    // //get ptr with specified offset
    // template<typename T>
    // inline T* getPtrWithOffset(const std::string& key, int index) const
    // {
    //     // TODO: add a check to check key is existed
    //     return tensor_map_.at(key).getPtrByOffset<T>(index);
    // }

    //for debug
    std::vector<std::string> keys() const
    {
        std::vector<std::string> key_names;
        for (auto& kv : tensor_map_) {
            key_names.push_back(kv.first);
        }
        return key_names;
    }
    // 打印出tensormap中的所有key
    std::string toString()
    {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for (size_t i = 0; i < tensor_map_.size(); ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < tensor_map_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
};
