#pragma once

#include "mat.h"
#include <random>

namespace zmat{

template<class _Ty, size_t Dim>
template<class _ResTy, class _Fn>
_ResTy Matrix<_Ty, Dim>::accumulate(_Fn func, _ResTy&& res) const{
    if(!is_valid()){
        throw zutil::error_invalid_use();
    }
    if(is_continuous()){
        for(auto it = raw_begin(); it != raw_end(); ++it)
            func(*it, res);
    }else{
        for(auto it = begin(); it != end(); ++it)
            func(*it, res);
    }
    return res;
}

template<class _Ty, size_t Dim>
template<class _ResTy, class _Fn>
_ResTy Matrix<_Ty, Dim>::accumulate(_Fn func) const{
    if(!is_valid()){
        throw zutil::error_invalid_use();
    }

    _ResTy res = front();

    if(is_continuous()){
        for(auto it = raw_begin() + 1; it != raw_end(); ++it)
            func(*it, res);
    }else{
        for(auto it = begin() + 1; it != end(); ++it)
            func(*it, res);
    }
    return res;
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::max() const-> _Ty{
    return accumulate<_Ty>([](const _Ty& ele, _Ty& mx){
        if(mx < ele) mx = ele;
    });
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::min() const-> _Ty{
    return accumulate<_Ty>([](const _Ty& ele, _Ty& mn){
        if(ele < mn) mn = ele;
    });  
}

template<class _Ty, size_t Dim>
template<class _ResTy>
auto Matrix<_Ty, Dim>::sum() const-> _ResTy{
    return accumulate<_ResTy>([](const _Ty& ele, _ResTy& sum){
        sum += ele;
    });
}

template<class _Ty, size_t Dim>
template<class _Tp, std::enable_if_t<std::is_integral_v<_Tp>, size_t> _>
auto Matrix<_Ty, Dim>::mean() const-> double{
    return mean<double>();
}

template<class _Ty, size_t Dim>
template<class _Tp, std::enable_if_t<!std::is_integral_v<_Tp>, size_t> _>
auto Matrix<_Ty, Dim>::mean() const-> _Tp{
    return static_cast<_Tp>(sum())/static_cast<_Tp>(size());
}

template<class _Ty, size_t Dim>
template<class _Fn, std::enable_if_t<std::is_invocable_r_v<bool, _Fn, const _Ty&>, size_t> _>
auto Matrix<_Ty, Dim>::count_if(_Fn cond) const -> size_t{
    return accumulate<size_t>([&cond](const _Ty& ele, size_t& res){
        res += cond(ele);
    }, 0);
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::count(const _Ty& val) const -> size_t{
    return count_if([&val](const _Ty& ele){return ele == val;});
}

template<class _Ty, size_t Dim>
template<class _Tp, std::enable_if_t<std::is_arithmetic_v<_Tp>, size_t> _>
auto Matrix<_Ty, Dim>::count_nonzero() const -> size_t{
    return count_if([](const _Ty& ele){return ele != static_cast<_Tp>(0);});
}

template<class _Ty, size_t Dim>
void Matrix<_Ty, Dim>::fill(const _Ty& val){
    *this <= val;
}

template<class _Ty, size_t Dim>
template<class _ResTy, class _Fn, std::enable_if_t<std::is_invocable_r_v<_ResTy, _Fn, const _Ty&>, size_t> _>
auto Matrix<_Ty, Dim>::maps(_Fn mapper) const->Matrix<_ResTy, Dim>{
    if(!is_valid())
        throw zutil::error_invalid_use();
    Matrix<_ResTy, Dim> res(_sizes);
    internal::mat_apply(*this, res, mapper);
    return res;
}

template<class _Ty, size_t Dim>
template<class _Fn, std::enable_if_t<std::is_invocable_v<_Fn, _Ty&>, size_t> _>
void Matrix<_Ty, Dim>::apply(_Fn op){
    if(!is_valid())
        throw zutil::error_invalid_use();
    if(is_continuous()){
        for(auto it = raw_begin(); it != raw_end(); ++it)
            op(*it);
    }else{
        for(auto it = begin(); it != end(); ++it)
            op(*it);
    }
}

template<class _Ty, size_t Dim>
template<class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _>
auto Matrix<_Ty, Dim>::zeros(Types ...sizes)-> self{
    static_assert((std::is_convertible_v<Types, size_t> && ...), "Index should be size type.");
    shape_t idx = {static_cast<size_t>(sizes)...};
    return self(idx, 0);
}

template<class _Ty, size_t Dim>
template<class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _>
auto Matrix<_Ty, Dim>::ones(Types ...sizes)-> self{
    static_assert((std::is_convertible_v<Types, size_t> && ...), "Index should be size type.");
    shape_t idx = {static_cast<size_t>(sizes)...};
    return self(idx, 1);
}

template<class _Ty, size_t Dim>
template<size_t _N, std::enable_if_t<(_N == 2) && (_N == Dim), size_t> _>
auto Matrix<_Ty, Dim>::eye(size_t siz)-> self{
    self res(siz, siz, 0);
    pointer ptr = res.start_ptr;
    for(size_t i = 0; i < siz; ++i){
        *ptr = 1;
        ptr += siz + 1;
    }
    return res;
}

template<class _Ty, size_t Dim>
template<class Rand, class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _>
auto Matrix<_Ty, Dim>::random(Rand destri, Types ...sizes)-> self{
    static_assert((std::is_convertible_v<Types, index_t> && ...), "Index should be size type.");
    shape_t idx = {static_cast<index_t>(sizes)...};
    self res(idx);
    std::random_device seed;
    std::mt19937 rd(seed());
    for(auto it = res.raw_begin(); it != raw_end(); ++it){
        *it = destri(rd);
    }
}

}//namespace zmat