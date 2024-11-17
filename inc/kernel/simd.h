#pragma once
#include"iter.h"
#include<stdint.h>
#include<type_traits>

namespace zmat{
namespace simd{

template<typename _Ty>
void vec_add(const _Ty* a, const _Ty* b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] + b[i];
}

template<typename _Ty>
void vec_add(const _Ty* a, const _Ty& b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] + b;
}

template<typename _Ty>
void vec_add(const _Ty& a, const _Ty* b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a + b[i];
}

template<typename _Ty>
void vec_sub(const _Ty* a, const _Ty* b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] - b[i];
}

template<typename _Ty>
void vec_sub(const _Ty* a, const _Ty& b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] - b;
}

template<typename _Ty>
void vec_sub(const _Ty& a, const _Ty* b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a - b[i];
}

template<typename _Ty>
void vec_mul(const _Ty* a, const _Ty* b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] * b[i];
}

template<typename _Ty>
void vec_mul(const _Ty* a, const _Ty& b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] * b;
}

template<typename _Ty>
void vec_div(const _Ty* a, const _Ty* b, _Ty* dst, size_t size){
    #pragma omp simd
    for(int i = 0; i < size; ++i)
        dst[i] = a[i] / b[i];
}

}; // namespace simd
}; // namespace zmat
