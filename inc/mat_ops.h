#pragma once

#include "mat.h"
#include "kernel/simd.h"
#include <iostream>
#include <cstdio>
#include <algorithm>

namespace zmat{

template<class _Ty, size_t Dim>
void Matrix<_Ty, Dim>::print(std::ostream& out, std::shared_ptr<formatter<_Ty>> nfmt) const{
    if(!is_valid()){
        out << "null";
        return;
    }
    if(!nfmt)
        nfmt = fmt;
    out << nfmt->st;

    size_t siz = size(0);

    if(siz > std::max(nfmt->max_items, nfmt->front_items + nfmt->back_items)){
        for(size_t i = 0; i < nfmt->front_items; ++i){
            if constexpr(Dim == 1){
                out << nfmt->to_string((*this)[i]);
            }else{
                (*this)[i].print(out, (nfmt->recursive? nfmt: nullptr));
            }
            out << nfmt->del;
        }

        out << "..." << nfmt->del;

        for(index_t i = -(index_t)nfmt->back_items; i < 0; ++i){
            if constexpr(Dim == 1){
                out << nfmt->to_string((*this)[i]);
            }else{
                (*this)[i].print(out, (nfmt->recursive? nfmt: nullptr));
            }
            if(i != - 1)
                out << nfmt->del;
        }
    }else{
        for(size_t i = 0; i < siz; ++i){
            if constexpr(Dim == 1){
                out << nfmt->to_string((*this)[i]);
            }else{
                (*this)[i].print(out, (nfmt->recursive? nfmt: nullptr));
            }
            if(i != siz - 1)
                out << nfmt->del;
        }
    }

    out << nfmt->ed;
}

template<class _Ty, size_t Dim>
std::ostream& operator <<(std::ostream& out, const Matrix<_Ty, Dim>& mat){
    mat.print(out);
    return out;
}

namespace internal{
template<class _T1, class _T2, class _Res, class _Fn, size_t Dim>
void mat_apply(const Matrix<_T1, Dim>& a, const Matrix<_T2, Dim>& b, Matrix<_Res, Dim>& res, _Fn func){
    if(a.is_continuous() && b.is_continuous()){
        std::transform(a.raw_begin(), a.raw_end(), b.raw_begin(), res.raw_begin(), func);
    }else if(a.is_continuous()){
        std::transform(a.raw_begin(), a.raw_end(), b.begin(), res.raw_begin(), func);
    }else if(b.is_continuous()){
        std::transform(a.begin(), a.end(), b.raw_begin(), res.raw_begin(), func);
    }else{
        std::transform(a.begin(), a.end(), b.begin(), res.raw_begin(), func);
    }
}

template<class _T1, class _Res, class _Fn, size_t Dim>
void mat_apply(const Matrix<_T1, Dim>& a, Matrix<_Res, Dim>& res, _Fn func){
    if(a.is_continuous())
        std::transform(a.raw_begin(), a.raw_end(), res.raw_begin(), func);
    else
        std::transform(a.begin(), a.end(), res.raw_begin(), func);
}

template<class _It1, class _It2>
bool _Mat_cmp_eps_n(_It1 st1, _It1 end, _It2 st2){
    for(; st1 != end; ++st1, ++st2){
        if(std::abs(*st1 - *st2) > mat_get_eps())
            return false;
    }
    return true;
}

template<class _It1, class _It2>
bool _Mat_cmp_n(_It1 st1, _It1 end, _It2 st2){
    for(; st1 != end; ++st1, ++st2){
        if(*st1 != *st2)
            return false;
    }
    return true;
}

template<class _T1, class _T2, size_t Dim>
bool mat_cmp_eps(const Matrix<_T1, Dim>& a, const Matrix<_T2, Dim>& b){
    if(a.is_continuous() && b.is_continuous()){
        return _Mat_cmp_eps_n(a.raw_begin(), a.raw_end(), b.raw_begin());
    }else if(a.is_continuous()){
        return _Mat_cmp_eps_n(a.raw_begin(), a.raw_end(), b.begin());
    }else if(b.is_continuous()){
        return _Mat_cmp_eps_n(b.raw_begin(), b.raw_end(), a.begin());
    }else{
        return _Mat_cmp_eps_n(a.begin(), a.end(), b.begin());
    }
}

template<class _T1, class _T2, size_t Dim>
bool mat_cmp(const Matrix<_T1, Dim>& a, const Matrix<_T2, Dim>& b){
    if(a.is_continuous() && b.is_continuous()){
        return _Mat_cmp_n(a.raw_begin(), a.raw_end(), b.raw_begin());
    }else if(a.is_continuous()){
        return _Mat_cmp_n(a.raw_begin(), a.raw_end(), b.begin());
    }else if(b.is_continuous()){
        return _Mat_cmp_n(b.raw_begin(), b.raw_end(), a.begin());
    }else{
        return _Mat_cmp_n(a.begin(), a.end(), b.begin());
    }
}

template<typename _Ty>
void gemm(const _Ty *a, const _Ty *b, _Ty* dst,
             const size_t M, const size_t K, const size_t N,
             const size_t step_a, const size_t step_b, const size_t step_dst){
    constexpr size_t BS = 1024 / sizeof(_Ty);
    constexpr size_t MAT_ALIGN = 16;

    alignas(MAT_ALIGN) _Ty a_buf[BS][BS];
    alignas(MAT_ALIGN) _Ty b_buf[BS][BS];
    alignas(MAT_ALIGN) _Ty res_buf[BS][BS] = {};

    for(size_t bi = 0; bi < M; bi += BS){
        size_t li = std::min(M - bi, BS);
        for(size_t bk = 0; bk < K; bk += BS){
            size_t lk = std::min(K - bk, BS);
            for(int i = 0; i < li; ++i){
                auto ptr = a + (i + bi) * step_a + bk;
                for(int k = 0; k < lk; ++k)
                    a_buf[i][k] = ptr[k];
            }

            for(size_t bj = 0; bj < N; bj += BS){
                size_t lj = std::min(N - bj, BS);

                for(int k = 0; k < lk; ++k){
                    auto ptr = b + (k + bk) * step_b + bj;
                    for(int j = 0; j < lj; ++j)
                        b_buf[k][j] = ptr[j];
                }

                for(int i = 0; i < li; ++i){
                    for(int k = 0; k < lk; ++k){
                        #pragma omp simd
                        for(int j = 0; j < lj; ++j){
                            res_buf[i][j] += a_buf[i][k] * b_buf[k][j];
                        }
                    }
                }

                for(int i = 0; i < li; ++i){
                    auto ptr = dst + (i + bi) * step_dst + bj;
                    for(int j = 0; j < lj; ++j){
                        ptr[j] += res_buf[i][j];
                        res_buf[i][j] = 0;
                    }
                }
            }
        }
    }
}

}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>:: operator <<=(const Matrix<_T, Dim>& mat)-> self&{
    if(!is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != mat._sizes)
        throw std::invalid_argument("shape mismatch");

    if(mat.is_continuous()){
        if(is_continuous())
            std::copy_n(mat.raw_begin(), mat.size(), raw_begin());
        else
            std::copy_n(mat.raw_begin(), mat.size(), begin());
    }else{
        if(is_continuous())
            std::copy_n(mat.begin(), mat.size(), raw_begin());
        else
            std::copy_n(mat.begin(), mat.size(), begin());
    }

    return *this;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>:: operator <<=(const _T& val)-> self&{

    if(!is_valid())
        throw zutil::error_invalid_use();

    if(is_continuous()){
        std::fill_n(raw_begin(), size(), val);
    }else{
        std::fill_n(begin(), size(), val);
    }

    return *this;
}

template<class _Ty, size_t Dim>
template<class _T>
Matrix<decltype(std::declval<_Ty>() + std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator +(const Matrix<_T, Dim>& b) const{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");
    
    Matrix<decltype(std::declval<_Ty>() + std::declval<_T>()), Dim> res(_sizes);
    internal::mat_apply(*this, b, res, std::plus<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _>
Matrix<decltype(std::declval<_Ty>() + std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator +(const _T& b) const{
    if(!is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_Ty>() + std::declval<_T>()), Dim> res(_sizes);
    internal::mat_apply(*this, res, [&b](const _Ty& val){return val + b;});
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator +=(const Matrix<_T, Dim>& b)-> self&{
    return *this <= *this + b;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator +=(const _T& b)-> self&{
    return *this <= *this + b;
}

template<class _T1, class _T2, size_t Dim, std::enable_if_t<!is_matrix_v<_T1>, size_t> _ = 0>
Matrix<decltype(std::declval<_T1>() + std::declval<_T2>()), Dim>
operator +(const _T1& a, const Matrix<_T2, Dim>& b){
    if(!b.is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_T1>() + std::declval<_T2>()), Dim> res(b._sizes);
    internal::mat_apply(b, res, [&a](const _T1& val){return a + val;});
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
Matrix<decltype(std::declval<_Ty>() - std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator -(const Matrix<_T, Dim>& b) const{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");    

    Matrix<decltype(std::declval<_Ty>() - std::declval<_T>()), Dim> res(_sizes);

    if constexpr(std::is_same_v<_Ty, _T> && std::is_arithmetic_v<_Ty>){
        if(is_continuous() && b.is_continuous()){
            simd::vec_sub(raw_begin(), b.raw_begin(), res.raw_begin(), size());
            return res;
        }
    }

    internal::mat_apply(*this, b, res, std::minus<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _>
Matrix<decltype(std::declval<_Ty>() - std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator -(const _T& b) const{
    if(!is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_Ty>() - std::declval<_T>()), Dim> res(_sizes);

    if constexpr(std::is_same_v<_Ty, _T> && std::is_arithmetic_v<_Ty>){
        if(is_continuous()){
            simd::vec_sub(raw_begin(), b, res.raw_begin(), size());
            return res;
        }
    }

    internal::mat_apply(*this, res, [&b](const _Ty& val){return val - b;});
    return res;
}

template<class _T1, class _T2, size_t Dim, std::enable_if_t<!is_matrix_v<_T1>, size_t> _ = 0>
Matrix<decltype(std::declval<_T1>() - std::declval<_T2>()), Dim>
operator -(const _T1& a, const Matrix<_T2, Dim>& b){
    if(!b.is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_T1>() - std::declval<_T2>()), Dim> res(b._sizes);
    internal::mat_apply(b, res, [&a](const _T1& val){return a - val;});
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator -=(const Matrix<_T, Dim>& b)-> self&{
    return *this <= *this - b;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator -=(const _T& b)-> self&{
    return *this <= *this - b;
}

template<class _Ty, size_t Dim>
template<size_t _N, std::enable_if_t<(_N == 2) && (_N == Dim), size_t> _>
auto Matrix<_Ty, Dim>::operator *(const Matrix<_Ty, _N>& b) const-> self{
    if(!is_valid()|| !b.is_valid())
        throw zutil::error_invalid_use();
    if(cols() != b.rows())
        throw std::invalid_argument("shape mismatch");
    
    size_t M = rows(), K = cols(), N = b.cols();
    Matrix<_Ty, 2> res(M, N);

    if constexpr(std::is_arithmetic_v<_Ty>){
        internal::gemm(start_ptr, b.start_ptr, res.start_ptr, 
                        M, K, N, step(0), b.step(0), 1);
    }else{
        for(size_t i = 0; i < M; ++i)
            for(size_t k = 0; k < K; ++k){
                _Ty tmp = at(i, k);
                pointer res_ptr = res.start_ptr + i * N;
                pointer b_ptr = b.start_ptr + k * b._steps[0];
                for(size_t j = 0; j < N; ++j){
                    *res_ptr++ += tmp * *b_ptr++;
                }
            }
    }
    return res;
}

template<class _Ty, size_t Dim>
template<size_t _N, std::enable_if_t<(_N == 2) && (_N == Dim), size_t> _>
auto Matrix<_Ty, Dim>::operator *(const Matrix<_Ty, 1>& b) const-> self{
    if(!is_valid()|| !b.is_valid())
        throw zutil::error_invalid_use();
    if(cols() != b.size())
        throw std::invalid_argument("shape mismatch");
    
    return *this * b.reinterpret(b.size(), 1);
}

template<class _Ty, size_t Dim>
template<size_t _N, std::enable_if_t<(_N == 1) && (_N == Dim), size_t> _>
auto Matrix<_Ty, Dim>::operator *(const Matrix<_Ty, _N>& b) const-> _Ty{
    if(!is_valid()|| !b.is_valid())
        throw zutil::error_invalid_use();
    if(size() != b.size())
        throw std::invalid_argument("shape mismatch");
    size_t siz = size();
    _Ty res = start_ptr[0] * b.start_ptr[0];
    for(size_t i = 1; i < siz; ++i)
        res += start_ptr[i] * b.start_ptr[i];
    return res;
}

template<class _Ty, size_t Dim>
template<size_t _N, std::enable_if_t<(_N == 1) && (_N == Dim), size_t> _>
auto Matrix<_Ty, Dim>::operator *(const Matrix<_Ty, 2>& b) const-> self{
    if(!is_valid()|| !b.is_valid())
        throw zutil::error_invalid_use();
    if(size() != b.rows())
        throw std::invalid_argument("shape mismatch");
    
    return (reinterpret(1, size()) * b).reinterpret(b.cols());
}

template<class _Ty, size_t Dim>
template<size_t _N, std::enable_if_t<(_N == 2) && (_N == Dim), size_t> _>
auto Matrix<_Ty, Dim>::operator *=(const self& b) const-> self&{
    return *this = *this * b;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator *=(const _T& b) const-> self&{
    return *this <= *this * b;
}

template<class _Ty, size_t Dim>
template<class _T>
Matrix<decltype(std::declval<_Ty>() * std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::mul(const Matrix<_T, Dim>& b) const{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");    

    Matrix<decltype(std::declval<_Ty>() * std::declval<_T>()), Dim> res(_sizes);

    if constexpr(std::is_same_v<_Ty, _T> && std::is_arithmetic_v<_Ty>){
        if(is_continuous() && b.is_continuous()){
            simd::vec_mul(raw_begin(), b.raw_begin(), res.raw_begin(), size());
            return res;
        }
    }

    internal::mat_apply(*this, b, res, std::multiplies<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _>
Matrix<decltype(std::declval<_Ty>() * std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator *(const _T& b) const{
    if(!is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_Ty>() * std::declval<_T>()), Dim> res(_sizes);

    if constexpr(std::is_same_v<_Ty, _T> && std::is_arithmetic_v<_Ty>){
        if(is_continuous()){
            simd::vec_mul(raw_begin(), b, res.raw_begin(), size());
            return res;
        }
    }

    internal::mat_apply(*this, res, [&b](const _Ty& val){return val * b;});
    return res;
}

template<class _T1, class _T2, size_t Dim, std::enable_if_t<!is_matrix_v<_T1>, size_t> _ = 0>
Matrix<decltype(std::declval<_T1>() * std::declval<_T2>()), Dim>
operator *(const _T1& a, const Matrix<_T2, Dim>& b){
    if(!b.is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_T1>() * std::declval<_T2>()), Dim> res(b._sizes);
    internal::mat_apply(b, res, [&a](const _T1& val){return a * val;});
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
Matrix<decltype(std::declval<_Ty>() / std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator /(const Matrix<_T, Dim>& b) const{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");    

    Matrix<decltype(std::declval<_Ty>() / std::declval<_T>()), Dim> res(_sizes);

    if constexpr(std::is_same_v<_Ty, _T> && std::is_arithmetic_v<_Ty>){
        if(is_continuous() && b.is_continuous()){
            simd::vec_div(raw_begin(), b.raw_begin(), res.raw_begin(), size());
            return res;
        }
    }

    internal::mat_apply(*this, b, res, std::divides<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _>
Matrix<decltype(std::declval<_Ty>() / std::declval<_T>()), Dim>
Matrix<_Ty, Dim>::operator /(const _T& b) const{
    if(!is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_Ty>() / std::declval<_T>()), Dim> res(_sizes);

    if constexpr(std::is_same_v<_Ty, _T> && std::is_arithmetic_v<_Ty>){
        if(is_continuous()){
            simd::vec_div(raw_begin(), b, res.raw_begin(), size());
            return res;
        }
    }

    internal::mat_apply(*this, res, [&b](const _Ty& val){return val / b;});
    return res;
}

template<class _T1, class _T2, size_t Dim, std::enable_if_t<!is_matrix_v<_T1>, size_t> _ = 0>
Matrix<decltype(std::declval<_T1>() / std::declval<_T2>()), Dim>
operator /(const _T1& a, const Matrix<_T2, Dim>& b){
    if(!b.is_valid())
        throw zutil::error_invalid_use();

    Matrix<decltype(std::declval<_T1>() / std::declval<_T2>()), Dim> res(b._sizes);
    internal::mat_apply(b, res, [&a](const _T1& val){return a / val;});
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator /=(const Matrix<_T, Dim>& b) const-> self&{
    return *this <= *this / b;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator /=(const _T& b) const-> self&{
    return *this <= *this / b;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator ==(const Matrix<_T, Dim>& b) const-> bool{
    if(!is_valid() || !b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        return false;

    if constexpr(std::is_floating_point_v<_Ty> || std::is_floating_point_v<_T>){
        return internal::mat_cmp_eps(*this, b);
    }else{
        return internal::mat_cmp(*this, b);
    }
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator !=(const Matrix<_T, Dim>& b) const-> bool{
    return !(*this == b);
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator <(const Matrix<_T, Dim>& b) const-> Matrix<bool, Dim>{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");

    Matrix<bool, Dim> res(_sizes);
    internal::mat_apply(*this, b, res, std::less<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator <=(const Matrix<_T, Dim>& b) const-> Matrix<bool, Dim>{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");

    Matrix<bool, Dim> res(_sizes);
    internal::mat_apply(*this, b, res, std::less_equal<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator >(const Matrix<_T, Dim>& b) const-> Matrix<bool, Dim>{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");

    Matrix<bool, Dim> res(_sizes);
    internal::mat_apply(*this, b, res, std::greater<>());
    return res;
}

template<class _Ty, size_t Dim>
template<class _T>
auto Matrix<_Ty, Dim>::operator >=(const Matrix<_T, Dim>& b) const-> Matrix<bool, Dim>{
    if(!is_valid()||!b.is_valid())
        throw zutil::error_invalid_use();
    if(_sizes != b._sizes)
        throw std::invalid_argument("shape mismatch");

    Matrix<bool, Dim> res(_sizes);
    internal::mat_apply(*this, b, res, std::greater_equal<>());
    return res;
}
    
} // namespace zmat
