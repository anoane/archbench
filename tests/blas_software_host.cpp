// SPDX-License-Identifier: AGPL-3.0-or-later
// Exercise the actual portable algorithms used by CUDA software BLAS. This is
// host algebra/indexing verification, not CUDA compilation or GPU validation.
#include "archbench/blas_software.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Z=std::complex<double>;
struct ops {
    using value=Z;
    static Z zero(){return {};}
    static Z one(){return 1.;}
    static Z add(Z a,Z b){return a+b;}
    static Z sub(Z a,Z b){return a-b;}
    static Z mul(Z a,Z b){return a*b;}
    static Z madd(Z a,Z b,Z c){return a*b+c;}
    static Z div(Z a,Z b){return a/b;}
    static Z conj(Z a){return std::conj(a);}
    static Z real_part(Z a){return {a.real(),0};}
};
namespace sb=archbench::software_blas;
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
void close(Z a,Z b,const char* text){require(std::abs(a-b)<2e-11*std::max(1.,std::abs(b)),text);}
Z sample(int row,int col,int seed){return {((row*7+col*3+seed)%17-8)/16.,((row*5-col*2+seed)%13)/32.};}
std::vector<Z> stored(int rows,int cols,int ld,int seed){
    std::vector<Z> out(std::size_t(ld)*cols,Z(999,999));
    for(int col=0;col<cols;++col)for(int row=0;row<rows;++row)out[std::size_t(col)*ld+row]=sample(row,col,seed);
    return out;
}
// Independent row-major materialization, used for dense reference operations.
std::vector<Z> dense(const std::vector<Z>& a,int rows,int cols,int ld,int trans){
    const int out_rows=trans?cols:rows,out_cols=trans?rows:cols;
    std::vector<Z> result(std::size_t(out_rows)*out_cols);
    for(int col=0;col<cols;++col)for(int row=0;row<rows;++row){
        const auto x=trans==2?std::conj(a[std::size_t(col)*ld+row]):a[std::size_t(col)*ld+row];
        result[trans?std::size_t(col)*out_cols+row:std::size_t(row)*out_cols+col]=x;
    }
    return result;
}
std::vector<Z> triangular_dense(const std::vector<Z>& a,int order,int ld,bool lower,bool unit,int trans){
    auto data=a;
    for(int col=0;col<order;++col)for(int row=0;row<order;++row){
        if((lower&&row<col)||(!lower&&row>col))data[std::size_t(col)*ld+row]=0;
        if(unit&&row==col)data[std::size_t(col)*ld+row]=1;
    }
    return dense(data,order,order,ld,trans);
}
void solve_tests(){
    constexpr int m=3,n=4,ldc=6;
    const Z alpha(.75,.125);
    int cases=0;
    for(bool left:{false,true})for(bool lower:{false,true})for(bool unit:{false,true})for(int trans:{0,1,2}){
        const int order=left?m:n,lda=order+2;
        auto a=stored(order,order,lda,3);
        for(int i=0;i<order;++i)a[std::size_t(i)*lda+i]=Z(2,.125);
        auto b=stored(m,n,ldc,11),solution=b;
        for(int rhs=0;rhs<(left?n:m);++rhs)sb::solve_rhs<ops>(rhs,m,n,trans,left,lower,unit,alpha,a.data(),lda,solution.data(),ldc);
        const auto matrix=triangular_dense(a,order,lda,lower,unit,trans);
        for(int row=0;row<m;++row)for(int col=0;col<n;++col){
            Z residual{};
            for(int p=0;p<order;++p)
                residual+=left?matrix[std::size_t(row)*order+p]*solution[std::size_t(col)*ldc+p]
                              :solution[std::size_t(p)*ldc+row]*matrix[std::size_t(p)*order+col];
            close(residual,alpha*b[std::size_t(col)*ldc+row],"triangular solve residual");
        }
        for(int col=0;col<n;++col)for(int row=m;row<ldc;++row)
            require(solution[std::size_t(col)*ldc+row]==Z(999,999),"triangular solve overwrote padding");
        ++cases;
    }
    require(cases==24,"missing solve controls");
}
void structure_tests(){
    constexpr int m=3,n=4,ldc=5;
    const Z alpha(.75,.125),beta(.25,-.125);
    int cases=0;
    for(auto op:{sb::operation::trmm,sb::operation::symm,sb::operation::hemm})
    for(bool left:{false,true})for(bool lower:{false,true})for(bool unit:{false,true})for(int trans:{0,1,2}){
        if(op!=sb::operation::trmm&&trans)continue;
        const int order=left?m:n,lda=order+2,ldb=m+3;
        auto a=stored(order,order,lda,4),b=stored(m,n,ldb,7),c=stored(m,n,ldc,10);
        std::vector<Z> matrix;
        if(op==sb::operation::trmm)matrix=triangular_dense(a,order,lda,lower,unit,trans);
        else {
            matrix.resize(std::size_t(order)*order);
            for(int col=0;col<order;++col)for(int row=0;row<order;++row){
                if((lower&&row<col)||(!lower&&row>col))continue;
                auto value=a[std::size_t(col)*lda+row];
                if(op==sb::operation::hemm&&row==col)value=Z(value.real(),0);
                matrix[std::size_t(row)*order+col]=value;
                matrix[std::size_t(col)*order+row]=op==sb::operation::hemm?std::conj(value):value;
            }
        }
        for(int row=0;row<m;++row)for(int col=0;col<n;++col){
            Z expected{};
            for(int p=0;p<order;++p)
                expected+=left?matrix[std::size_t(row)*order+p]*b[std::size_t(col)*ldb+p]
                              :b[std::size_t(p)*ldb+row]*matrix[std::size_t(p)*order+col];
            expected*=alpha;if(op!=sb::operation::trmm)expected+=beta*c[std::size_t(col)*ldc+row];
            const auto actual=sb::matrix_element<ops>(op,row,col,m,n,0,trans,0,left,lower,unit,
                    alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc,nullptr,1,1);
            close(actual,expected,"structured matrix multiply");
        }
        ++cases;
    }
    require(cases==40,"missing structure controls");
}
void general_tests(){
    constexpr int m=3,n=4,k=5;
    const Z alpha(.75,.125),beta(.25,-.125);
    for(int ta:{0,1,2})for(int tb:{0,1,2}){
        const int ar=ta?n:m,ac=ta?m:n,br=tb?n:m,bc=tb?m:n;
        auto a=stored(ar,ac,ar+2,1),b=stored(br,bc,br+1,2),c=stored(m,n,m+2,3);
        const auto av=dense(a,ar,ac,ar+2,ta),bv=dense(b,br,bc,br+1,tb);
        for(int row=0;row<m;++row)for(int col=0;col<n;++col)
            close(sb::matrix_element<ops>(sb::operation::geam,row,col,m,n,k,ta,tb,true,true,false,
                    alpha,a.data(),ar+2,b.data(),br+1,beta,c.data(),m+2,nullptr,1,1),
                  alpha*av[std::size_t(row)*n+col]+beta*bv[std::size_t(row)*n+col],"GEAM transposition or conjugation");
    }
    for(int ta:{0,1,2}){
        const int length=ta?m:n,outputs=ta?n:m;
        auto a=stored(m,n,m+2,4),c=stored(1,outputs,3,6),x=stored(1,length,2,8);
        const auto av=dense(a,m,n,m+2,ta);
        for(int row=0;row<outputs;++row){
            Z expected=beta*c[std::size_t(row)*3];
            for(int p=0;p<length;++p)expected+=alpha*av[std::size_t(row)*length+p]*x[std::size_t(p)*2];
            close(sb::matrix_element<ops>(sb::operation::gemv,row,0,m,n,k,ta,0,true,true,false,
                    alpha,a.data(),m+2,nullptr,1,beta,c.data(),3,x.data(),2,3),expected,"GEMV strides or conjugation");
        }
    }
    for(int ta:{0,1}){
        const int rows=ta?k:n,cols=ta?n:k,lda=rows+2;
        auto a=stored(rows,cols,lda,5),c=stored(n,n,n+1,9);
        const auto av=dense(a,rows,cols,lda,ta);
        for(int row=0;row<n;++row)for(int col=0;col<n;++col){
            Z expected=beta*c[std::size_t(col)*(n+1)+row];
            for(int p=0;p<k;++p)expected+=alpha*av[std::size_t(row)*k+p]*av[std::size_t(col)*k+p];
            close(sb::matrix_element<ops>(sb::operation::syrk,row,col,n,n,k,ta,0,true,true,false,
                    alpha,a.data(),lda,nullptr,1,beta,c.data(),n+1,nullptr,1,1),expected,"SYRK uses symmetric transpose, not Hermitian");
        }
    }
}
int main(int argc,char**argv){
    try{
        require(argc==2,"scenario required");const std::string scenario=argv[1];
        if(scenario=="solve")solve_tests();
        else if(scenario=="structure")structure_tests();
        else if(scenario=="general")general_tests();
        else throw std::runtime_error("unknown scenario");
        std::cout<<"PASS "<<scenario<<"\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
