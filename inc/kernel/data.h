#pragma once

#include<memory>    //shared_ptr, allocator
#include<cstring>   //memcpy
#include<vector>
#include<iostream>

namespace zmat{

namespace internal{

template<class _Ty>
struct MatrixData;

template<>
struct MatrixData<void>{
    virtual ~MatrixData(){}
    
    virtual std::shared_ptr<MatrixData<void>> clone() const = 0;
    void* get_data() const {
        return data;
    }

protected:   
    void* data;
};

using data_manager = std::shared_ptr<MatrixData<void>>;

template<class _Ty>
struct MatrixData: public MatrixData<void>{
    using value_type = _Ty;
    using pointer = _Ty *;

private:
    using self = MatrixData<_Ty>;

    size_t size;

public:
    template<class ...Types>
    MatrixData(size_t size, Types&& ...args):size(size){
        std::allocator<_Ty> alloc;
        data = reinterpret_cast<void*>(alloc.allocate(size));
        _construct_in_range(get_data(), get_data() + size, std::forward<Types>(args)...);
    }

    MatrixData(){
        data = nullptr;
    }

    MatrixData(size_t size, const _Ty* src): size(size){
        if(src == nullptr)
            throw std::runtime_error("copy from a null pointer");
        std::allocator<_Ty> alloc;
        data = reinterpret_cast<void*>(alloc.allocate(size));
        std::copy(src, src + size, get_data());
    }

    MatrixData(size_t size, _Ty* src, bool clone = true): size(size){
        if(src == nullptr)
            throw std::runtime_error("copy from a null pointer");
        if(clone){
            std::allocator<_Ty> alloc;
            data = reinterpret_cast<void*>(alloc.allocate(size));
            std::copy(src, src + size, get_data());
        }else{
            data = src;
        }
    }

    ~MatrixData(){
        if(data != nullptr){
            std::allocator<_Ty> alloc;
            std::destroy_n(get_data(), size);
            alloc.deallocate(get_data(), size);
        }
    }

    pointer get_data() const{
        return reinterpret_cast<pointer>(data);
    }

    data_manager clone() const override{
        return std::make_shared<self>(size, get_data());
    }

    void allocate_uninitialized(size_t size){
        this->size = size;
        std::allocator<_Ty> alloc;
        data = reinterpret_cast<void*>(alloc.allocate(size));
    }

private:
    template<class ...Types>
    void _construct_in_range(pointer L, pointer R, Types&& ...args) const{
        std::allocator<_Ty> alloc;
        for(auto ptr = L; ptr != R; ++ptr){
            alloc.construct(ptr, std::forward<Types>(args)...);
        }
    }
};

template<class _Ty, class ...Types>
data_manager make_manager(Types ...args){
    return std::make_shared<MatrixData<_Ty>>(std::forward<Types>(args)...);
}

template<class _Ty>
data_manager make_manager_uninit(size_t size){
    auto res = std::make_shared<MatrixData<_Ty>>();
    res->allocate_uninitialized(size);
    return res;
}

};//namespace internal

};//namespace zmat