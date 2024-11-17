#include "matrix.h"
#include<algorithm>
#include<complex>
using namespace std;
using zmat::Matrix;
using zmat::Mat;
using zmat::Vector;

static int mov, cpy, cons, des;

template<class T>
T get_mid(T& a,T& b,T& c){
	if(a<b)
		return a<c?c:a;
	else return b<c?c:b;
}

template<class T>
void qksort(T s,T t){
    using Tp = typename T::value_type;
	if(t-s<=1)return;
	T lt=s,gt=t,p=s;
	Tp pv=get_mid(*s,*(t-1),*(s+(t-s)/2));
	while(p!=gt){
		if(*p==pv){
			++p;
			continue;
		}
		if(*p<pv){
			swap(*lt,*p);
			++lt;
			++p;
		}else{
			gt--;
			swap(*gt,*p);
		}
	}
	qksort(s,lt);
	qksort(gt,t);
}

class test{
    int a, b;
public:
    test(int a = 0){
        ++cons;
    }

    test(const test&& a){
        ++mov;
    }

    test(const test& a){
        ++cpy;
    }

    test& operator = (const test& a){
        ++cpy;
        return *this;
    }

    test& operator = (const test&& a){
        ++mov;
        return *this;
    }

    ~test(){
        ++des;
    }
};

void clear_cnt(){
    mov = cpy = cons = des = 0;
}

void show_cnt(){
    cout << "move: " << mov << ", copy: " << cpy << ", construct: " << cons << ", destroy: " << des << endl;
}

template<class T>
void print_msg(T a){

    cout << "size: " << a.size() << endl;
    cout << "dim: " << a.dims() << endl;
    cout << "sizes: [";
    for(int i = 0; i < a.dims(); ++i){
        cout << a.size(i) << ((i + 1 != a.dims())? ", ": "]\n");
    }
    cout << "steps: [";
    for(int i = 0; i < a.dims(); ++i){
        cout << a.step(i) << ((i + 1 != a.dims())? ", ": "]\n");
    }
    cout << boolalpha;
    cout << "continuous: " << a.is_continuous() << endl;
    cout << "view: " << a.is_view() << endl;
}

#include<ctime>
int main(){
#define PRINT(a) cout << #a << ": " << a << endl;
#define CATCH_ANY\
    catch(std::exception& e){\
        cout << "exception: " << e.what() << endl;\
    }

    {
        cout << "**************part1 info************" << endl;
        cout << "Matrix<int, 3> a({2, 3, 4}):" << endl;
        Matrix<int, 3> a({2, 3, 4});
        print_msg(a);

        cout << endl << "auto b = a[1]" << endl;
        auto b = a[1];
        print_msg(b);

        cout << endl << "c = a[:][1:][1-3]" << endl;
        auto c = a.view({{}, {1}, {1, 3}});
        print_msg(c);
    }

    {
        cout << "**************part2 initialization************" << endl;
        cout << "init size using std::vector." << endl;
        std::vector<size_t> siz = {2, 2};
        Mat<int> a(siz, 3);
        PRINT(a);

        cout << "init size using literal number." << endl;
        Vector<int> b(2, 3);
        Mat<string> c(1, 2, "e");
        PRINT(b);
        PRINT(c);

        cout << "init size using initializer-list." << endl;
        Matrix<int, 3> d({3, 2, 1});
        PRINT(d);

        cout << "array-like initialization." << endl;
        Vector<int> e = {1, 2, 3};
        Mat<double> f = {{1.0, 2.0} , {3.0}};
        Matrix<int, 3> g = {{{1, 2}, {3}}, {{4}, {5}, {6}}};
        PRINT(e);
        PRINT(f);
        PRINT(g);

        cout << "data copy initialization." << endl;
        int* h_dat = new int[4]{1, 2, 3, 4};
        Mat<int> h(2, 2, h_dat);
        PRINT(h);

        cout << "exceptions" << endl;
        try{
            Vector<int> a(0);
            PRINT(a);
        }CATCH_ANY;

        try{
            Matrix<int, 10> a({1, 2, 3, 4, 5});
            PRINT(a);
        }CATCH_ANY

        try{
            int* data = nullptr;
            Matrix<int, 2> a(1, 1, data);
            PRINT(a);
        }CATCH_ANY
    }

    {
        cout << "**************part3 assign************" << endl;
        Mat<test> a(3, 3);
        clear_cnt();
        auto b = a;
        cout << "after copy construction (using \'=\')" << endl;
        show_cnt();

        Mat<test> c(3, 3);
        clear_cnt();
        c <<= a;
        cout << "after assign (using \'<<=\')" << endl;
        show_cnt();

        clear_cnt();
        b = a.clone();
        cout << "after clone (using \'clone\')" << endl;
        show_cnt();
        clear_cnt();

        cout << "exceptions" << endl;
        try{
            Mat<int> a;
            auto b = a.clone();
            cout << "case1: ok" << endl;
        }CATCH_ANY;

        try{
            Mat<int> a;
            auto b = a;
            cout << "case2: ok" << endl;
        }CATCH_ANY;

        try{
            Mat<int> a(2, 2), b(2, 3);
            b <<= a;
            cout << "case3: ok" << endl;
        }CATCH_ANY;
    }

    {
        cout << "**************part4 access&view************" << endl;
        Matrix<int, 3> a = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
        PRINT(a);
        cout << "a[0]: " << a[0] << endl;
        cout << "a[1][1]: " << a[1][1] << endl;
        a.at(0, 1, 0) = -1;
        a[-1][-1] <<= -2;
        cout << "after assign: " << a << endl;

        a = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
        auto b = a.view({{}, {1}, {0, -1}});
        cout << endl << "b = a[0-1][1-1][0-1]" << endl;
        PRINT(b);
        cout << "b[0] <<= 1" << endl;
        b <<= 1;
        PRINT(b);
        PRINT(a);

        cout << "exceptions" << endl;
        try{
            a[10];
        }CATCH_ANY;
        try{
            a.view({2, 1});
        }CATCH_ANY;
    }

    {
        cout << "**************part5 iterator************" << endl;
        Mat<int> a = {{1, 2, 3}, {4, 5, 6}};
        PRINT(a);
        for(auto it = a.begin(); it < a.end(); it += 2)
            cout <<*it<<" ";
        cout << endl;
        for(auto& val: a)
            val = 1;
        PRINT(a);

        Mat<int> b = {{6, 5, 4}, {3, 2, 1}};
        sort(b.begin(), b.end());
        cout << b << endl;

        cout << "exceptions" << endl;
        try{
            Mat<int>::iterator a;
            ++a;
        }CATCH_ANY;
        try{
            Mat<int> a(2, 2);
            auto it = a.begin();
            --it;
            *it;
        }CATCH_ANY;
    }

    {
        cout << "**************part6 Operation************" << endl;
        Mat<int> a = {{1, 1, 4}, {5, 1, 4}};
        Mat<double> b = {{1, 9, 1}, {9, 8, 1}};
        PRINT(a);
        PRINT(b);

        PRINT(a + b);
        PRINT(a - b);
        PRINT(a.mul(b));
        PRINT(a / b);
        PRINT(a + 1);
        PRINT(1 - a);

        Mat<int> c = {{1, 2}, {3, 4}, {5, 6}};
        Vector<int> d = {1, 2, 3};
        Vector<int> e = {1, 2};
        PRINT(c);
        PRINT(d);
        PRINT(e);

        PRINT(a * c);
        PRINT(a * d);
        PRINT(e * a);
        PRINT(d * d);

        PRINT((a <= b));
        PRINT((Mat<double>(2, 2, 1e-8) == Mat<double>::zeros(2, 2)));
        PRINT((Mat<double>(2, 2, 1e-10) == Mat<double>::zeros(2, 2)));

        cout << "exceptions" << endl;
        try{
            auto x = a * e;
        }CATCH_ANY
    }

    {
        cout << "**************part7 Function************" << endl;
        Mat<pair<int, int>> a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        auto c = a.reinterpret<int>(2, 2, 2);
        PRINT(c);
        c.reshape(1, 2, 4);
        PRINT(c);

        PRINT(c[0].transposed());

        auto d = a.maps<double>([](const pair<int, int>& x){
            return (x.first + x.second) / 2.0;
        });

        PRINT(d);

        c.apply([](int& x){
            x *= x;
        });

        PRINT(c);
        PRINT(c.sum());
        PRINT(c.mean<double>());
        PRINT((c - 1).count_nonzero());

        cout << "exceptions" << endl;

        try{
            c[0].transpose();
        }CATCH_ANY

        try{
            a.reinterpret<int>(2, 2);
        }CATCH_ANY

        try{
            c.reshape(2, 2, 4);
        }CATCH_ANY
    }

    {
        cout << "**************part8 Different Types************" << endl;
        Mat<string> a(3, 3, "a");
        Mat<string> b(3, 3, "b");

        PRINT(a + b);

        Mat<complex<double>> c(2, 2, 1, -1);
        Mat<complex<double>> d(2, 2, 2, 0.5);

        PRINT(c);
        PRINT(d);
        PRINT(c * d);
    }
}