#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace zmat{

namespace zutil{

std::out_of_range error_out_of_range(ptrdiff_t index, ptrdiff_t limit);

std::logic_error error_invalid_use();

template<class ...Types>
std::string as_str(Types ...args){
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
}
    
} //namespace zutil

namespace internal{

template<class _Ty, size_t Dim>
struct init_val_base{
    std::vector<init_val_base<_Ty, Dim - 1>> ch;
    
    init_val_base(const std::initializer_list<init_val_base<_Ty, Dim - 1>>& args){
        for(const auto &val: args){
            ch.push_back(val);
        }
    }
};

template<class _Ty>
struct init_val_base<_Ty, 1>{

    std::vector<_Ty> ch;

    init_val_base(const std::initializer_list<_Ty>& args){
        for(const auto &val: args){
            ch.push_back(val);
        }
    }
};

template<class _Ty, size_t _N, class _It>
void collect_shape(const init_val_base<_Ty, _N>& base, _It dst){
    if constexpr(_N == 1){
        *dst = std::max(*dst, base.ch.size());
    }else{
        *dst = std::max(*dst, base.ch.size());
        ++dst;
        for(const auto &ch: base.ch)
            collect_shape(ch, dst);
    }
}

struct mat_setting{
    static double eps;
};

};//namespace internal


struct Range{
    ptrdiff_t l, r;

    Range(ptrdiff_t l = 0, ptrdiff_t r = -1): l(l), r(r){}
};

void mat_set_eps(double eps);
double mat_get_eps();

};//namespace zmat