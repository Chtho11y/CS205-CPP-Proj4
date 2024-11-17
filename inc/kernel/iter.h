#pragma once

#include<vector>
#include "utils.h"
#include<array>
#include<iostream>

namespace zmat{

template<size_t Dim>
using shape_type = std::array<size_t, Dim>;

template<class _Ty, size_t Dim>
class MatIterator{
public:
    using value_type = _Ty;
    using pointer = _Ty*;
    using reference = _Ty&;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = ptrdiff_t;

private:
    using shape_t = shape_type<Dim>;
    using self = MatIterator<_Ty, Dim>;

    shape_t _sizes, _steps;
    std::array<ptrdiff_t, Dim> idx;

    pointer base, _ptr;

    void move_ptr(ptrdiff_t diff){
        valid_check();
        if constexpr(Dim == 1){
            _ptr += diff * static_cast<ptrdiff_t>(_steps[0]);
            idx[0] += diff;
        }else{
            int carry = 0;
            for(size_t i = Dim - 1; i > 0; --i){
                ptrdiff_t siz = _sizes[i];
                ptrdiff_t step = _steps[i];
                ptrdiff_t cur = diff % siz;

                _ptr -= step * idx[i];
                idx[i] += cur;
                if(idx[i] < 0){
                    idx[i] += siz;
                    carry = -1;
                }
                if(idx[i] >= siz){
                    idx[i] -= siz;
                    carry = 1;
                }
                _ptr += step * idx[i];
                diff = diff / siz + carry;
                carry = 0;
                if(diff == 0)
                    return;
            }

            _ptr += diff * static_cast<ptrdiff_t>(_steps[0]);
            idx[0] += diff;
        }
    }

    void valid_check() const{
        if(_ptr == nullptr)
            throw std::logic_error("using iterator without initialization.");
    }

    void range_check() const{
        if(idx[0] < 0 || idx[0] >= _sizes[0])
            throw std::out_of_range("access an out-of-range iterator.");
    }

public:

    pointer data() const{
        return _ptr;
    }

    const int* index() const{
        return idx.data();
    }

    /*debug*/
    void print() const{
        for(auto i: idx)
            std::cout<<i<<" ";
        std::cout << std::endl;
    }

    MatIterator():
    base(nullptr), _ptr(nullptr){}

    MatIterator(pointer base, pointer ptr, const shape_t& size, const shape_t& step):
    base(base), _ptr(ptr), _sizes(size), _steps(step){
        ptrdiff_t diff = ptr - base;
        idx[0] = diff / static_cast<ptrdiff_t>(_steps[0]);
        diff %= static_cast<ptrdiff_t>(_steps[0]);
        if(diff < 0){
            diff += static_cast<ptrdiff_t>(_steps[0]);
            idx[0]--;
        }
        for(size_t i = 1; i < Dim; ++i){
            idx[i] = diff / static_cast<ptrdiff_t>(_steps[i]);
            diff %= static_cast<ptrdiff_t>(_steps[i]);
        }
    }

    MatIterator(const self& it) = default;
    self& operator = (const self& it) = default;

    _Ty& operator*() const{
        valid_check();
        range_check();

        return *_ptr;
    }

    _Ty* operator->() const{
        valid_check();
        range_check();
        return _ptr;
    }

    _Ty& operator[](ptrdiff_t diff) const{
        return *(*this + diff);
    }

    self operator++(){
        self res(*this);
        move_ptr(1);
        return res;
    }

    self& operator++(int){
        move_ptr(1);
        return *this;
    }

    self operator--(){
        self res(*this);
        move_ptr(-1);
        return res;
    }

    self& operator--(int){
        move_ptr(-1);
        return *this;
    }

    self operator +(ptrdiff_t idx) const{
        self res(*this);
        res.move_ptr(idx);
        return res;
    }

    self operator -(ptrdiff_t idx) const{
        self res(*this);
        res.move_ptr(-idx);
        return res;
    }

    ptrdiff_t operator -(const self& it) const{
        if(_sizes != it._sizes || _steps != it._steps || base != it.base)
            throw std::invalid_argument("iterator mismatch");
        ptrdiff_t res = 0, tot = 1;

        for(long long i = Dim - 1; i >= 0; --i){
            auto diff = idx[i] - it.idx[i];
            res += diff * tot;
            tot *= _sizes[i];
        }
        return res;
    }

    self& operator +=(ptrdiff_t idx){
        move_ptr(idx);
        return *this;
    }

    self& operator -=(ptrdiff_t idx){
        move_ptr(-idx);
        return *this;
    }

    bool operator<(const self& it) const{
        return _ptr < it._ptr;
    }

    bool operator<=(const self& it) const{
        return _ptr <= it._ptr;
    }

    bool operator>(const self& it) const{
        return _ptr > it._ptr;
    }

    bool operator>=(const self& it) const{
        return _ptr >= it._ptr;
    }

    bool operator==(const self& it) const{
        return _ptr == it._ptr;
    }

    bool operator!=(const self& it) const{
        return _ptr != it._ptr;
    }
};

template<class _Ty, size_t Dim>
MatIterator<_Ty, Dim> operator +(ptrdiff_t diff, MatIterator<_Ty, Dim> it){
    return it + diff;
}

template<class _Ty, size_t Dim>
using MatConstIterator = MatIterator<const _Ty, Dim>;

}