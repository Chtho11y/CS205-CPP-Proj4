#pragma once

#include "kernel/iter.h"
#include "kernel/data.h"
#include "kernel/utils.h"
#include "kernel/formatter.h"

namespace zmat{

enum MatFlag{
    CONTINUOUS_FLAG = 0x1, VIEW_FLAG = 0x2
};

template<class _Ty, size_t Dim>
class Matrix;

template<typename _Ty>
struct is_matrix : std::false_type {};

template<typename _Ele, size_t _Dim>
struct is_matrix<Matrix<_Ele, _Dim>> : std::true_type {};

/*test if a type is Matrix type.*/
template<typename _Ty>
constexpr bool is_matrix_v = is_matrix<std::decay_t<_Ty>>::value;

template<class _Ty, size_t Dim>
class Matrix{
    static_assert(Dim >= 1, "Dimension could not less than 1");
#define _MAT_DIM_RESTRICT(cond)\
    size_t _N = Dim, std::enable_if_t<(cond) && (_N == Dim), size_t> _ = 0

public:
    using value_type = _Ty;
    using pointer = _Ty*;
    using reference = _Ty&;

    using flag_t = uint32_t;
    using index_t = ptrdiff_t; //type used in operator[] and 'at'.

    using iterator = MatIterator<_Ty, Dim>;
    using const_iterator = MatConstIterator<_Ty, Dim>;

    using sub_type = std::conditional_t<Dim == 1, reference, Matrix<_Ty, Dim - 1>>;
    using shape_t = shape_type<Dim>;
    
    template<size_t _N>
    using sub_type_of = std::conditional_t<Dim == _N, reference, Matrix<_Ty, Dim - _N>>;

private:
    using data_manager = internal::data_manager;
    using def_fmt = std::conditional_t<internal::is_printable_v<_Ty>, default_fmt<_Ty>, formatter<_Ty>>;
    using self = Matrix<_Ty, Dim>;

    data_manager _raw_data;
    std::shared_ptr<formatter<_Ty>> fmt = std::make_shared<def_fmt>();

    shape_t _sizes, _steps;

    flag_t flag;

    pointer start_ptr;

    template<class _It, class ..._Args>
    void init_shape(_It arg, _Args ...args);

    template<class _It1, class _It2>
    Matrix(pointer st_ptr, data_manager raw, _It1 shape_it, _It2 step_it);

    void recalc_continuous();

    template<class _ResTy, class _Fn>
    _ResTy accumulate(_Fn func, _ResTy&& res) const;
    template<class _ResTy, class _Fn>
    _ResTy accumulate(_Fn func) const;

    template<class _It>
    void bind(_It shape, pointer ptr);

public:

    Matrix();

    Matrix(const self &mat);
    Matrix(self &&mat);

    template<class ..._ArgTy>
    Matrix(const std::vector<size_t>& sizes, _ArgTy ...args);

    template<class ..._ArgTy>
    Matrix(const shape_t& sizes, _ArgTy ...args);

    template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 1)>
    explicit Matrix(const size_t siz, _ArgTy ...args);

    template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 2)>
    Matrix(const size_t rw, const size_t cl, _ArgTy ...args);

    template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N >= 2)>
    Matrix(const std::initializer_list<size_t>& size, _ArgTy ...args);

    template<_MAT_DIM_RESTRICT(_N >= 2)>
    Matrix(const std::initializer_list<internal::init_val_base<_Ty, Dim - 1>>& init_vals);

    template<_MAT_DIM_RESTRICT(_N == 1)>
    Matrix(const std::initializer_list<_Ty>& init_vals);

    template<_MAT_DIM_RESTRICT(_N >= 2)>
    Matrix(pointer ptr, const std::initializer_list<size_t>& size);

    Matrix(pointer ptr, const std::vector<size_t>& sizes);

    template<_MAT_DIM_RESTRICT(_N == 2)>
    Matrix(pointer ptr, const size_t x, const size_t y);

    template<_MAT_DIM_RESTRICT(_N == 1)>
    Matrix(pointer ptr, const size_t x);

    template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 1)>
    void create(const size_t siz, _ArgTy ...args);

    template<class ..._ArgTy>
    void create(const std::vector<size_t>& sizes, _ArgTy ...args);

    template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N == 2)>
    void create(const size_t rw, const size_t cl, _ArgTy ...args);

    template<class ..._ArgTy, _MAT_DIM_RESTRICT(_N >= 2)>
    void create(const std::initializer_list<size_t>& size, _ArgTy ...args);

    template<_MAT_DIM_RESTRICT(_N >= 2)>
    void create(pointer ptr, const std::initializer_list<size_t>& size);

    void create(pointer ptr, const std::vector<size_t>& sizes);

    template<_MAT_DIM_RESTRICT(_N == 2)>
    void create(pointer ptr, const size_t x, const size_t y);

    template<_MAT_DIM_RESTRICT(_N == 1)>
    void create(pointer ptr, const size_t x);

    template<_MAT_DIM_RESTRICT(_N >= 2)>
    void create(const std::initializer_list<internal::init_val_base<_Ty, Dim - 1>>& init_vals);

    template<_MAT_DIM_RESTRICT(_N == 1)>
    void create(const std::initializer_list<_Ty>& init_vals);

    void reset();

    size_t size(size_t index) const;
    size_t size() const;
    size_t step(size_t index) const;
    constexpr size_t dims() const;

    template<_MAT_DIM_RESTRICT(_N == 2)>
    size_t cols() const;
    template<_MAT_DIM_RESTRICT(_N == 2)>
    size_t rows() const;

    flag_t get_flag() const;
    bool is_continuous() const;
    bool is_view() const;
    bool is_valid() const;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    const _Ty* raw_begin() const;
    const _Ty* raw_end() const;

    _Ty* raw_begin();
    _Ty* raw_end();

    _Ty& front();
    _Ty& back();

    const _Ty& front() const;
    const _Ty& back() const;   

    sub_type operator[](index_t);
    const sub_type operator[](index_t) const;

    template<class ...Types, std::enable_if_t<(sizeof...(Types) <= Dim), size_t> _ = 0>
    sub_type_of<sizeof...(Types)> at(Types ...indices);

    template<class ...Types, std::enable_if_t<(sizeof...(Types) <= Dim), size_t> _ = 0>
    const sub_type_of<sizeof...(Types)> at(Types ...indices) const;

    void print(std::ostream& out, std::shared_ptr<formatter<_Ty>> fmt = nullptr) const;

    self& operator =(const self &mat);
    self& operator =(self &&mat);
    self clone() const;

    self view(index_t l_bound, index_t r_bound);
    template<_MAT_DIM_RESTRICT(_N == 2)>
    self view(index_t top, index_t bottom, index_t left, index_t right);
    self view(const std::initializer_list<Range>& rngs);

    const self view(index_t l_bound, index_t r_bound) const;
    template<_MAT_DIM_RESTRICT(_N == 2)>
    const self view(index_t top, index_t bottom, index_t left, index_t right) const;
    const self view(std::initializer_list<Range> rngs) const;

    template<_MAT_DIM_RESTRICT(_N == 2)>
    self row_view(index_t idx);
    template<_MAT_DIM_RESTRICT(_N == 2)>
    const self row_view(index_t idx) const;
    template<_MAT_DIM_RESTRICT(_N == 2)>
    self col_view(index_t idx);
    template<_MAT_DIM_RESTRICT(_N == 2)>
    const self col_view(index_t idx) const;

    template<_MAT_DIM_RESTRICT(_N == 2)>
    void transpose();
    template<_MAT_DIM_RESTRICT(_N <= 2)>
    Matrix<_Ty, 2> transposed() const;

    template<class _T>
    Matrix<decltype(std::declval<_Ty>() * std::declval<_T>()), Dim>
    mul(const Matrix<_T, Dim>& b) const;
    
    template<class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _ = 0>
    void reshape(Types ...args);

    template<class _Tp = _Ty, class ...Types>
    Matrix<_Tp, sizeof...(Types)> reinterpret(Types ...args) const;

    template<class _Fn, std::enable_if_t<std::is_invocable_v<_Fn, _Ty&>, size_t> _ = 0>
    void apply(_Fn op);
    template<class _Res, class _Fn, std::enable_if_t<std::is_invocable_r_v<_Res, _Fn, const _Ty&>, size_t> _ = 0>
    Matrix<_Res, Dim> maps(_Fn mapper) const;

    void fill(const _Ty&);
    template<class _T>
    self& operator <<=(const Matrix<_T, Dim> &mat);
    template<class _T>
    self& operator <<=(const _T& val);

    template<class _T>
    Matrix<decltype(std::declval<_Ty>() + std::declval<_T>()), Dim>
    operator +(const Matrix<_T, Dim>&) const;

    template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _ = 0>
    Matrix<decltype(std::declval<_Ty>() + std::declval<_T>()), Dim>
    operator +(const _T&) const;

    template<class _T1, class _T2, size_t _N, std::enable_if_t<!is_matrix_v<_T1>, size_t> _>
    friend Matrix<decltype(std::declval<_T1>() + std::declval<_T2>()), _N>
    operator +(const _T1&, const Matrix<_T2, _N>&);

    template<class _T>
    self& operator +=(const Matrix<_T, Dim>&);
    template<class _T>
    self& operator +=(const _T&);

    template<class _T>
    Matrix<decltype(std::declval<_Ty>() - std::declval<_T>()), Dim>
    operator -(const Matrix<_T, Dim>&) const;

    template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _ = 0>
    Matrix<decltype(std::declval<_Ty>() - std::declval<_T>()), Dim>
    operator -(const _T&) const;

    template<class _T1, class _T2, size_t _N, std::enable_if_t<!is_matrix_v<_T1>, size_t> _>
    friend Matrix<decltype(std::declval<_T1>() - std::declval<_T2>()), _N>
    operator -(const _T1&, const Matrix<_T2, _N>&);

    template<class _T>
    self& operator -=(const Matrix<_T, Dim>&);
    template<class _T>
    self& operator -=(const _T&);

    template<_MAT_DIM_RESTRICT(_N == 2)>
    self operator *(const Matrix<_Ty, _N>&) const;
    template<_MAT_DIM_RESTRICT(_N == 2)>
    self operator *(const Matrix<_Ty, 1>&) const;
    template<_MAT_DIM_RESTRICT(_N == 2)>
    self& operator *=(const self&) const;

    template<_MAT_DIM_RESTRICT(_N == 1)>
    _Ty operator *(const Matrix<_Ty, _N>&) const;
    template<_MAT_DIM_RESTRICT(_N == 1)>
    self operator *(const Matrix<_Ty, 2>&) const;


    template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _ = 0>
    Matrix<decltype(std::declval<_Ty>() * std::declval<_T>()), Dim>
    operator *(const _T&) const;

    template<class _T>
    self& operator *=(const _T&) const;

    template<class _T1, class _T2, size_t _N, std::enable_if_t<!is_matrix_v<_T1>, size_t> _>
    friend Matrix<decltype(std::declval<_T1>() * std::declval<_T2>()), _N>
    operator *(const _T1&, const Matrix<_T2, _N>&);

    template<class _T>
    Matrix<decltype(std::declval<_Ty>() / std::declval<_T>()), Dim>
    operator /(const Matrix<_T, Dim>&) const;

    template<class _T, std::enable_if_t<!is_matrix_v<_T>, size_t> _ = 0>
    Matrix<decltype(std::declval<_Ty>() / std::declval<_T>()), Dim>
    operator /(const _T&) const;

    template<class _T1, class _T2, size_t _N, std::enable_if_t<!is_matrix_v<_T1>, size_t> _>
    friend Matrix<decltype(std::declval<_T1>() / std::declval<_T2>()), _N>
    operator /(const _T1&, const Matrix<_T2, _N>&);

    template<class _T>
    self& operator /=(const _T&) const;

    template<class _T>
    self& operator /=(const Matrix<_T, Dim>&) const;

    template<class _T>
    Matrix<bool, Dim> operator <=(const Matrix<_T, Dim>&) const;
    template<class _T>
    Matrix<bool, Dim> operator >=(const Matrix<_T, Dim>&) const;
    template<class _T>
    Matrix<bool, Dim> operator <(const Matrix<_T, Dim>&) const;
    template<class _T>
    Matrix<bool, Dim> operator >(const Matrix<_T, Dim>&) const;

    template<class _T>
    bool operator ==(const Matrix<_T, Dim>&) const;
    template<class _T>
    bool operator !=(const Matrix<_T, Dim>&) const;

    _Ty max() const;
    _Ty min() const;

    template<class _ResTy = _Ty>
    _ResTy sum() const;

    template<class _Tp = _Ty, std::enable_if_t<std::is_integral_v<_Tp>, size_t> _ = 0>
    double mean() const;
    template<class _Tp = _Ty, std::enable_if_t<!std::is_integral_v<_Tp>, size_t> _ = 0>
    _Tp mean() const;

    template<class _Tp = _Ty, std::enable_if_t<std::is_arithmetic_v<_Tp>, size_t> _ = 0>
    size_t count_nonzero() const;

    size_t count(const _Ty& val) const;
    template<class _Fn, std::enable_if_t<std::is_invocable_r_v<bool, _Fn, const _Ty&> , size_t> _ = 0>
    size_t count_if(_Fn cond) const;

    template<class _Other, size_t _N>
    friend class Matrix;

    template<class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _ = 0>
    static self zeros(Types ...sizes);
    template<class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _ = 0>
    static self ones(Types ...sizes);
    template<_MAT_DIM_RESTRICT(_N == 2)>
    static self eye(size_t siz);
    template<class Rand, class ...Types, std::enable_if_t<(sizeof...(Types) == Dim), size_t> _ = 0>
    static self random(Rand destri, Types ...sizes); 

#undef _MAT_DIM_RESTRICT
};

template<class _Ty>
using Vector = Matrix<_Ty, 1>;

template<class _Ty>
using Mat = Matrix<_Ty, 2>;


};// namespace zmat