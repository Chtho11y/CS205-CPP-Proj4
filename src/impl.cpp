#include "kernel/utils.h"

namespace zmat{

namespace zutil{

std::out_of_range error_out_of_range(ptrdiff_t index, ptrdiff_t limit){
    return std::out_of_range("index " + std::to_string(index) + " exceed the size " + std::to_string(limit));
}

std::logic_error error_invalid_use(){
    return std::logic_error("using an uninitialized matrix");
}

}// namespace zutil

namespace internal{
double mat_setting::eps = 1e-9;
}

void mat_set_eps(double eps){
    internal::mat_setting::eps = eps;
}

double mat_get_eps(){
    return internal::mat_setting::eps;

    
} // namespace internal
}; //namespace zmat