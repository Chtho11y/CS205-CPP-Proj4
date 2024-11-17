#pragma once

#include<string>
#include<sstream>
#include<type_traits>
#include<functional>

namespace zmat{

namespace internal{

template<typename T>
struct is_printable {
private:

    template<typename U>
    static auto test(U* ptr) -> decltype(std::declval<std::ostream&>() << std::declval<U>(), std::true_type());

    template<typename>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template<class _Ty>
constexpr bool is_printable_v = is_printable<_Ty>::value;

} // namespace internal


template<class _Ty>
struct formatter{
    size_t max_items = 6;   //maximum # of display item.
    size_t front_items = 3;
    size_t back_items = 3;

    std::string del = ", "; //delimiter
    std::string st = "[";  //str before all items
    std::string ed = "]";  //str after all items

    bool recursive = false;
    bool binary = false;

    virtual std::string to_string(const _Ty& val){
        return "undefined";
    };
};

template<class _Ty>
struct default_fmt: formatter<_Ty>{

    std::string to_string(const _Ty& val) override{
        std::stringstream ss;
        ss << val;
        return ss.str();
    };
};

template<class _Ty>
struct float_fmt: formatter<_Ty>{
static_assert(std::is_floating_point_v<_Ty>, "Requiring floating point type in float_fmt.");

    std::string fmt;

    float_fmt(std::string fmt = ""): fmt(fmt){}

    std::string to_string(const _Ty& val){
        return "TODO";
    }
};

template<class _Ty>
struct integer_fmt: formatter<_Ty>{
static_assert(std::is_integral_v<_Ty>, "Requiring floating point type in float_fmt.");

    std::string fmt;

    integer_fmt(std::string fmt = ""): fmt(fmt){}

    std::string to_string(const _Ty& val){
        return "TODO";
    }
};

template<class _Ty>
struct custom_fmt: formatter<_Ty>{

    std::function<std::string(_Ty)> func;

    custom_fmt(std::function<std::string(_Ty)> func):func(func){};

    std::string to_string(const _Ty& val){
        return func(val);
    }
};

} // namespace zmat
