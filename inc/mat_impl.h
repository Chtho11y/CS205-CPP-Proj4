#pragma once

#include "mat.h"

namespace zmat{

namespace internal{

#define _MAT_DIM_RESTRICT(cond)\
    size_t _N, std::enable_if_t<(cond) && (_N == Dim), size_t> _

template<size_t Dim, class Iter>
void set_size_and_step(std::array<size_t, Dim>& sizes, std::array<size_t, Dim>& steps, Iter src){
    for(size_t i = 0; i < Dim; ++i)
        sizes[i] = *src++;
    
    steps[Dim - 1] = 1;
    for(size_t i = Dim - 1; i > 0; --i)
        steps[i - 1] = steps[i] * sizes[i];
}

template<class _SrcIt, class _Ty>
void copy_construct_uninit(_Ty* _begin, _SrcIt _src, size_t size){
    std::allocator<_Ty> alloc;
    for(size_t i = 0; i < size; ++i){
        alloc.construct(_begin, *_src);
        ++_begin;
        ++_src;
    }
}

template<size_t Dim, class _DstIt, class _ShIt, class _Ty>
void fill_init_value(_DstIt dst, _ShIt size_it, const init_val_base<_Ty, Dim>& init_vals){
    const auto& ch = init_vals.ch;
    if constexpr(Dim == 1){
        for(size_t i = 0; i < ch.size(); ++i){
            dst[i] = ch[i];
        }
    }else{
        size_t step = *size_it;
        ++size_it;
        for(size_t i = 0; i < ch.size(); ++i){
            fill_init_value<Dim - 1>(dst + i * step, size_it, ch[i]);
        }
    }
}

template<size_t Dim, class _It, class _Ptr>
shape_type<Dim> set_view_config(_It rng_it, const size_t siz, const shape_type<Dim>& _sizes,
                            const shape_type<Dim>& _steps, _Ptr& st_ptr){

    shape_type<Dim> nsiz = _sizes;

    for(size_t i = 0; i < siz; ++i){
        auto [l, r] = *rng_it++;

        if(l < 0) l += _sizes[i];
        if(r < 0) r += _sizes[i];
        if(l > r)
            throw std::invalid_argument(zutil::as_str("left bound ", l, " exceeds the right bound ", r));
        
        if(l < 0)
            throw std::out_of_range("left bound less than 0");
        if(r >= _sizes[i])
            throw zutil::error_out_of_range(r, _sizes[i]);
        nsiz[i] = r - l + 1;
        st_ptr += l * _steps[i];
    }
    return nsiz;
}

} // namespace internal


template<class _Ty, size_t Dim>
template<class _It, class ..._Args>
void Matrix<_Ty, Dim>::init_shape(_It it, _Args ...args){

    auto tmp = it;
    for(size_t i = 0; i < Dim; ++i){
        if(*tmp++ == 0)
            throw std::invalid_argument("matrix size cannot be zero.");
    }

    internal::set_size_and_step(_sizes, _steps, it);
    size_t siz = size();

    _raw_data = internal::make_manager<_Ty>(siz, std::forward<_Args>(args)...);
    start_ptr = reinterpret_cast<_Ty*>(_raw_data->get_data());
    flag = CONTINUOUS_FLAG;
}

template<class _Ty, size_t Dim>
void Matrix<_Ty, Dim>::recalc_continuous(){
    bool flag = (_steps[Dim - 1] == 1);
    for(size_t i = Dim - 1; (i > 0) && flag; --i){
        if(_sizes[i] * _steps[i] != _steps[i - 1]){
            flag = false;
            break;
        }
    }

    if(flag != is_continuous())
        this->flag ^= CONTINUOUS_FLAG;
}
template<class _Ty, size_t Dim>
template<class _It>
void Matrix<_Ty, Dim>::bind(_It shape, pointer ptr){
    reset();
    init_shape(shape, ptr, false);
}

template<class _Ty, size_t Dim> 
Matrix<_Ty, Dim>::Matrix(){
    reset();
}

template<class _Ty, size_t Dim> 
Matrix<_Ty, Dim>::Matrix(const self& mat){
    *this = mat;
}

template<class _Ty, size_t Dim> 
Matrix<_Ty, Dim>::Matrix(self&& mat){
    *this = mat;
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 1)>
void Matrix<_Ty, Dim>::create(const size_t siz, _ArgTy ...args) {
    reset();
    shape_t arr = {siz};
    init_shape(arr.begin(), std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 2)>
void Matrix<_Ty, Dim>::create(const size_t rw, const size_t cl, _ArgTy ...args){
    reset();
    shape_t siz = {rw, cl};
    init_shape(siz.begin(), std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N >= 2)>
void Matrix<_Ty, Dim>::create(pointer ptr, const std::initializer_list<size_t>& size){
    bind(size.begin(), ptr);
}

template<class _Ty, size_t Dim>
void Matrix<_Ty, Dim>::create(pointer ptr, const std::vector<size_t>& sizes){
    bind(sizes.begin(), ptr);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
void Matrix<_Ty, Dim>::create(pointer ptr, const size_t rw, const size_t cl){
    shape_t siz = {rw, cl};
    bind(siz.begin(), ptr);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 1)>
void Matrix<_Ty, Dim>::create(pointer ptr, const size_t x){
    shape_t siz = {x};
    bind(siz.begin(), ptr);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy>
void Matrix<_Ty, Dim>::create(const std::vector<size_t>& sizes, _ArgTy ...args){
    if(sizes.size() != Dim){
        throw std::invalid_argument(zutil::as_str("Dimension mismatch: dim = ", Dim, ", provided ", sizes.size()));
    }
    reset();
    init_shape(sizes.begin(), std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N >= 2)>
void Matrix<_Ty, Dim>::create(const std::initializer_list<size_t>& sizes, _ArgTy ...args){
    if(sizes.size() != Dim){
        throw std::invalid_argument(zutil::as_str("Dimension mismatch: dim = ", Dim, ", provided ", sizes.size()));
    }
    reset();
    init_shape(sizes.begin(), std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N >= 2)>
void Matrix<_Ty, Dim>::create(const std::initializer_list<internal::init_val_base<_Ty, Dim - 1>>& init_vals){
    reset();
    shape_t siz;
    siz.fill(0);
    siz[0] = init_vals.size();
    for(const auto& val: init_vals)
        internal::collect_shape(val, siz.begin() + 1);

    init_shape(siz.begin());
    size_t i = 0;
    for(const auto& val: init_vals){
        internal::fill_init_value<Dim - 1>(start_ptr + i * _steps[0], _steps.begin() + 1, val);
        ++i;   
    }
    flag = CONTINUOUS_FLAG;
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 1)>
void Matrix<_Ty, Dim>::create(const std::initializer_list<_Ty>& init_vals){

    if(init_vals.size() == 0){
        throw std::invalid_argument("matrix size cannot be zero.");
    }

    reset();

    _steps[0] = 1;
    _sizes[0] = init_vals.size();

    _raw_data = internal::make_manager_uninit<_Ty>(size());
    start_ptr = reinterpret_cast<_Ty*>(_raw_data->get_data());

    internal::copy_construct_uninit(start_ptr, init_vals.begin(), _sizes[0]);

    flag = CONTINUOUS_FLAG;
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 1)>
Matrix<_Ty, Dim>::Matrix(const size_t siz, _ArgTy ...args) {
    create(siz, std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 2)>
Matrix<_Ty, Dim>::Matrix(const size_t rw, const size_t cl, _ArgTy ...args){
    create(rw, cl, std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy>
Matrix<_Ty, Dim>::Matrix(const std::vector<size_t>& sizes, _ArgTy ...args){
    create(sizes, std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy>
Matrix<_Ty, Dim>::Matrix(const shape_t& sizes, _ArgTy ...args){
    reset();
    init_shape(sizes.begin(), std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim> 
template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N >= 2)>
Matrix<_Ty, Dim>::Matrix(const std::initializer_list<size_t>& sizes, _ArgTy ...args){
    create(sizes, std::forward<_ArgTy>(args)...);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N >= 2)>
Matrix<_Ty, Dim>::Matrix(const std::initializer_list<internal::init_val_base<_Ty, Dim - 1>>& init_vals){
    create(init_vals);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 1)>
Matrix<_Ty, Dim>::Matrix(const std::initializer_list<_Ty>& init_vals){
    create(init_vals);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N >= 2)>
Matrix<_Ty, Dim>::Matrix(pointer ptr, const std::initializer_list<size_t>& size){
    bind(size.begin(), ptr);
}

template<class _Ty, size_t Dim>
Matrix<_Ty, Dim>::Matrix(pointer ptr, const std::vector<size_t>& sizes){
    bind(sizes.begin(), ptr);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
Matrix<_Ty, Dim>::Matrix(pointer ptr, const size_t x, const size_t y){
    create(ptr, x, y);
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 1)>
Matrix<_Ty, Dim>::Matrix(pointer ptr, const size_t x){
    create(ptr, x);
}

template<class _Ty, size_t Dim> 
template<class _It1, class _It2>
Matrix<_Ty, Dim>::Matrix(pointer st_ptr,data_manager raw, _It1 shape_it, _It2 step_it):
_raw_data(raw), start_ptr(st_ptr){

    for(size_t i = 0; i < Dim; ++i){
        _sizes[i] = *shape_it++;
        _steps[i] = *step_it++;
    }

    flag = VIEW_FLAG;
    recalc_continuous();
}

template<class _Ty, size_t Dim> 
void Matrix<_Ty, Dim>:: reset(){
    _raw_data = nullptr;
    start_ptr = nullptr;
    flag = 0;
    _sizes.fill(0);
    _steps.fill(0);
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: size() const-> size_t{
    if constexpr(Dim == 1){
        return _sizes[0];
    }else{
        if(is_continuous())
            return step(0) * size(0);
        size_t tot = 1;
        for(auto i: _sizes)
            tot *= i;
        return tot;
    }
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: size(size_t index) const-> size_t{
    return _sizes[index];
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: step(size_t index) const-> size_t{
    return _steps[index];
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: get_flag() const-> flag_t{
    return flag;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: is_continuous() const-> bool{
    return flag & CONTINUOUS_FLAG;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: is_view() const-> bool{
    return flag & VIEW_FLAG;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: is_valid() const-> bool{
    return (bool)_raw_data;
}

template<class _Ty, size_t Dim> 
constexpr auto Matrix<_Ty, Dim>:: dims() const-> size_t{
    return Dim;
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>:: rows() const-> size_t{
    return _sizes[0];
}

template<class _Ty, size_t Dim> 
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>:: cols() const-> size_t{
    return _sizes[1];
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: operator[](index_t idx)-> sub_type{
    if(!is_valid())
        throw zutil::error_invalid_use();
    if(idx < 0)
        idx += size(0);
    if(idx >= size(0) || idx < 0)
        throw zutil::error_out_of_range(idx, size(0));
    if constexpr (Dim == 1){
        return *(start_ptr + idx * step(0));
    }else{
        return sub_type(start_ptr + idx * step(0), _raw_data, _sizes.begin() + 1, _steps.begin() + 1);
    }
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: operator[](index_t idx) const-> const sub_type{
    if(!is_valid())
        throw zutil::error_invalid_use();
    if(idx < 0)
        idx += size(0);
    if(idx >= size(0) || idx < 0)
        throw zutil::error_out_of_range(idx, size(0));
    if constexpr (Dim == 1){
        return *(start_ptr + idx * step(0));
    }else{
        return sub_type(start_ptr + idx * step(0), _raw_data, _sizes.begin() + 1, _steps.begin() + 1);
    }
}

template<class _Ty, size_t Dim> 
template<class ...Types, std::enable_if_t<(sizeof...(Types) <= Dim), size_t> _>
auto Matrix<_Ty, Dim>:: at(Types ...indices)-> sub_type_of<sizeof...(Types)>{
    constexpr size_t arg_cnt = sizeof...(Types);
    static_assert((std::is_convertible_v<Types, index_t> && ...), "Index should be size type.");

    if(!is_valid())
        throw zutil::error_invalid_use();

    index_t idx[] = {static_cast<index_t>(indices)...};

    auto ptr = start_ptr;

    for(size_t i = 0; i < arg_cnt; ++i){
        if(idx[i] < 0)
            idx[i] += size(i);
        if(idx[i] >= size(i) || idx[i] < 0)
            throw zutil::error_out_of_range(idx[i], size(i));
        ptr += idx[i] * step(i);
    }

    if constexpr(arg_cnt == Dim){
        return *ptr;
    }else{
        return sub_type_of<arg_cnt>(ptr, _raw_data, _sizes.begin() + arg_cnt, _steps.begin() + arg_cnt);
    }
}

template<class _Ty, size_t Dim> 
template<class ...Types, std::enable_if_t<(sizeof...(Types) <= Dim), size_t> _>
auto Matrix<_Ty, Dim>:: at(Types ...indices) const-> const sub_type_of<sizeof...(Types)>{
    constexpr size_t arg_cnt = sizeof...(Types);
    static_assert((std::is_convertible_v<Types, index_t> && ...), "Index should be size type.");

    if(!is_valid())
        throw zutil::error_invalid_use();

    index_t idx[] = {static_cast<index_t>(indices)...};

    auto ptr = start_ptr;

    for(size_t i = 0; i < arg_cnt; ++i){
        if(idx[i] < 0)
            idx[i] += size(i);
        if(idx[i] >= size(i) || idx[i] < 0)
            throw zutil::error_out_of_range(idx[i], size(i));
        ptr += idx[i] * step(i);
    }

    if constexpr(arg_cnt == Dim){
        return *ptr;
    }else{
        return sub_type_of<arg_cnt>(ptr, _raw_data, _sizes.begin() + arg_cnt, _steps.begin() + arg_cnt);
    }
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::view(index_t l, index_t r)-> self{
    return view({{l, r}});
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>::view(index_t top, index_t bottom, index_t left, index_t right)-> self{
    return view({{top, bottom}, {left, right}});
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::view(const std::initializer_list<Range>& rngs)-> self{
    if(rngs.size() > Dim){
        throw std::invalid_argument("ranges provided exceed the dimension");
    }

    auto st_ptr = start_ptr;
    auto res = internal::set_view_config(rngs.begin(), rngs.size(), _sizes, _steps, st_ptr);
    return self(st_ptr, _raw_data, res.begin(), _steps.begin());
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::view(index_t l, index_t r) const-> const self{
    return view({{l, r}});
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>::view(index_t top, index_t bottom, index_t left, index_t right) const-> const self{
    return view({{top, bottom}, {left, right}});
}

template<class _Ty, size_t Dim>
auto Matrix<_Ty, Dim>::view(std::initializer_list<Range> rngs) const-> const self{
    if(rngs.size() > Dim){
        throw std::invalid_argument("ranges provided exceed the dimension");
    }

    if(!is_valid())
        throw zutil::error_invalid_use();

    auto st_ptr = start_ptr;
    auto res = internal::set_view_config(rngs.begin(), rngs.size(), _sizes, _steps, st_ptr);
    return self(st_ptr, _raw_data, res.begin(), _steps.begin());
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>::col_view(index_t idx)-> self{
    return view({{}, {idx, idx}});
}


template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>::col_view(index_t idx) const-> const self{
    return view({{}, {idx, idx}});
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>::row_view(index_t idx)-> self{
    return view({{idx, idx}, {}});
}


template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
auto Matrix<_Ty, Dim>::row_view(index_t idx) const-> const self{
    return view({{idx, idx}, {}});
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: operator =(const self& mat)-> self&{
    _sizes = mat._sizes;
    _steps = mat._steps;
    _raw_data = mat._raw_data;
    start_ptr = mat.start_ptr;
    flag = mat.flag | VIEW_FLAG;
    return *this;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: operator =(self&& mat)-> self&{
    _sizes = std::move(mat._sizes);
    _steps = std::move(mat._steps);
    _raw_data = mat._raw_data;
    start_ptr = mat.start_ptr;
    flag = mat.flag;

    mat._raw_data = nullptr;
    mat.start_ptr = nullptr;
    mat.flag = 0;
    return *this;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: clone() const-> self{
    if(!is_valid())
        throw zutil::error_invalid_use();

    self res;

    res.flag = CONTINUOUS_FLAG;
    internal::set_size_and_step(res._sizes, res._steps, _sizes.begin());

    if(is_continuous()){
        res._raw_data = internal::make_manager<_Ty>(res.size(), raw_begin());
        res.start_ptr = reinterpret_cast<_Ty*>(res._raw_data->get_data());
    }else{
        res._raw_data = internal::make_manager_uninit<_Ty>(res.size());
        res.start_ptr = reinterpret_cast<_Ty*>(res._raw_data->get_data());

        internal::copy_construct_uninit(res.start_ptr, begin(), res.size());
    }

    return res;
}

template<class _Ty, size_t Dim> 
template<class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _>
void Matrix<_Ty, Dim>::reshape(Types ...args){
    static_assert((std::is_convertible_v<Types, size_t> && ...), "Index should be size type.");

    if(!is_valid())
        throw zutil::error_invalid_use();
    if(!is_continuous())
        throw std::logic_error("cannot reshape a non-continuous matrix");

    size_t sizes[] = {static_cast<size_t>(args)...};
    size_t tot = 1;
    for(auto siz: sizes)
        tot *= siz;
    if(tot != size())
        throw std::invalid_argument("Matrix size cannot change in reshape.");
    
    internal::set_size_and_step(_sizes, _steps, sizes);
}

template<class _Ty, size_t Dim> 
template<class _Tp, class ...Types>
auto Matrix<_Ty, Dim>::reinterpret(Types ...args) const-> Matrix<_Tp, sizeof...(Types)>{
    constexpr size_t arg_cnt = sizeof...(Types);
    static_assert((std::is_convertible_v<Types, size_t> && ...), "Index should be size type.");

    if(!is_valid())
        throw zutil::error_invalid_use();
    if(!is_continuous())
        throw std::logic_error("cannot reinterpret a non-continuous matrix");

    size_t sizes[] = {static_cast<size_t>(args)...};
    size_t tot = 1;
    for(auto siz: sizes)
        tot *= siz;
    
    if(tot * sizeof(_Tp) != size() * sizeof(_Ty))
        throw std::invalid_argument("Matrix size cannot change in reinterpret.");

    Matrix<_Tp, arg_cnt> res;

    res._raw_data = _raw_data;
    res.start_ptr = reinterpret_cast<_Tp*>(start_ptr);
    internal::set_size_and_step(res._sizes, res._steps, sizes);
    res.flag = flag;
    return res;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: begin()-> iterator{
    return iterator(start_ptr, start_ptr, _sizes, _steps);
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: end()-> iterator{
    return iterator(start_ptr, start_ptr + size(0) * step(0), _sizes, _steps);
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: begin() const-> const_iterator{
    return const_iterator(start_ptr, start_ptr, _sizes, _steps);
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: end() const-> const_iterator{
    return const_iterator(start_ptr, start_ptr + size(0) * step(0), _sizes, _steps);
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: raw_begin() const-> const _Ty*{
    return start_ptr;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: raw_end() const-> const _Ty*{
    if(!is_continuous()){
        throw std::logic_error("cannot get the raw end pointer of a non-continuous matrix");
    }
    return start_ptr + size();
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: front()-> _Ty&{
    return *raw_begin();
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: back()-> _Ty&{
    return *--end();
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: front() const-> const _Ty&{
    return *raw_begin();
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: back() const-> const _Ty&{
    return *--end();
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: raw_begin()-> pointer{
    return start_ptr;
}

template<class _Ty, size_t Dim> 
auto Matrix<_Ty, Dim>:: raw_end()-> pointer{
    if(!is_continuous()){
        throw std::logic_error("cannot get the raw end pointer of a non-continuous matrix");
    }
    return start_ptr + size();
}

template<class _Ty, size_t Dim>
template<_MAT_DIM_RESTRICT(_N == 2)>
void Matrix<_Ty, Dim>::transpose(){
    if(is_view()){
        if(cols() != rows()){
            throw std::logic_error("cannot change the layout when transposing a matrix view.");
        }
        for(size_t i = 0; i < rows(); ++i)
            for(size_t j = 0; j < i; ++j)
                std::swap(at(i, j), at(j, i));
    }else{
        *this = transposed();
    }
}

template<class _Ty, size_t Dim> 
template<_MAT_DIM_RESTRICT(_N <= 2)>
auto Matrix<_Ty, Dim>::transposed() const-> Matrix<_Ty, 2>{
    if constexpr(Dim == 2){
        self res(cols(), rows());
        for(size_t i = 0; i < rows(); ++i)
            for(size_t j = 0; j < cols(); ++j)
                res.at(j, i) = at(i, j);
        return res;
    }else{
        return clone().reinterpret(size(), 1);
    }
}

#undef _MAT_DIM_RESTRICT
} // namespace zmat