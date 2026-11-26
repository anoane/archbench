// SPDX-License-Identifier: AGPL-3.0-or-later
// Included inside gemm_detail after real_scalar<T>. Pure indexing/solve logic is
// in archbench/blas_software.hpp, included at global scope by gemm.cu.
template<class T> struct sb_ops {
    using value = T;
    using real = T;
    __host__ __device__ static T zero() { return T(0); }
    __host__ __device__ static T one() { return T(1); }
    __host__ __device__ static T add(T a,T b) { return a+b; }
    __host__ __device__ static T sub(T a,T b) { return a-b; }
    __host__ __device__ static T mul(T a,T b) { return a*b; }
    __host__ __device__ static T madd(T a,T b,T c) { return a*b+c; }
    __host__ __device__ static T div(T a,T b) { return a/b; }
    __host__ __device__ static T conj(T a) { return a; }
    __host__ __device__ static T real_part(T a) { return a; }
    __host__ __device__ static T abs1(T a) { return a < 0 ? -a : a; }
    __host__ __device__ static T norm2(T a) { return a*a; }
};
template<> struct sb_ops<cuComplex> {
    using value=cuComplex; using real=float;
    __host__ __device__ static value zero(){return make_cuComplex(0,0);}
    __host__ __device__ static value one(){return make_cuComplex(1,0);}
    __host__ __device__ static value add(value a,value b){return cuCaddf(a,b);}
    __host__ __device__ static value sub(value a,value b){return cuCsubf(a,b);}
    __host__ __device__ static value mul(value a,value b){return cuCmulf(a,b);}
    __host__ __device__ static value madd(value a,value b,value c){return add(mul(a,b),c);}
    __host__ __device__ static value div(value a,value b){return cuCdivf(a,b);}
    __host__ __device__ static value conj(value a){return cuConjf(a);}
    __host__ __device__ static value real_part(value a){return make_cuComplex(a.x,0);}
    __host__ __device__ static real abs1(value a){return fabsf(a.x)+fabsf(a.y);}
    __host__ __device__ static real norm2(value a){return a.x*a.x+a.y*a.y;}
};
template<> struct sb_ops<cuDoubleComplex> {
    using value=cuDoubleComplex; using real=double;
    __host__ __device__ static value zero(){return make_cuDoubleComplex(0,0);}
    __host__ __device__ static value one(){return make_cuDoubleComplex(1,0);}
    __host__ __device__ static value add(value a,value b){return cuCadd(a,b);}
    __host__ __device__ static value sub(value a,value b){return cuCsub(a,b);}
    __host__ __device__ static value mul(value a,value b){return cuCmul(a,b);}
    __host__ __device__ static value madd(value a,value b,value c){return add(mul(a,b),c);}
    __host__ __device__ static value div(value a,value b){return cuCdiv(a,b);}
    __host__ __device__ static value conj(value a){return cuConj(a);}
    __host__ __device__ static value real_part(value a){return make_cuDoubleComplex(a.x,0);}
    __host__ __device__ static real abs1(value a){return fabs(a.x)+fabs(a.y);}
    __host__ __device__ static real norm2(value a){return a.x*a.x+a.y*a.y;}
};
using sb_operation=::archbench::software_blas::operation;

inline sb_operation sb_parse_op(const std::string& name) {
#define SB_OP(op) if(name==#op)return sb_operation::op
    SB_OP(axpy);SB_OP(scal);SB_OP(dot);SB_OP(nrm2);SB_OP(asum);SB_OP(iamax);SB_OP(iamin);
    SB_OP(copy);SB_OP(swap);SB_OP(gemv);SB_OP(geam);SB_OP(trsm);SB_OP(trmm);
    SB_OP(symm);SB_OP(hemm);SB_OP(syrk);
#undef SB_OP
    throw std::invalid_argument("unknown software BLAS operation: "+name);
}
inline int sb_transpose(cublasOperation_t op) {
    return op==CUBLAS_OP_N?0:op==CUBLAS_OP_T?1:2;
}

template<class T> __global__ void sb_vector_kernel(sb_operation op,int n,T alpha,T* x,int ix,T* y,int iy) {
    using O=sb_ops<T>;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x) {
        const auto xi=i*ix,yi=i*iy;
        if(op==sb_operation::axpy)y[yi]=O::madd(alpha,x[xi],y[yi]);
        else if(op==sb_operation::scal)x[xi]=O::mul(alpha,x[xi]);
        else if(op==sb_operation::copy)y[yi]=x[xi];
        else if(op==sb_operation::swap){const T old=x[xi];x[xi]=y[yi];y[yi]=old;}
    }
}

template<class T> __global__ void sb_reduction_kernel(sb_operation op,int n,const T* x,int ix,
        const T* y,int iy,T* output,bool conjugate) {
    using O=sb_ops<T>;using R=typename O::real;
    __shared__ T sums[256];
    __shared__ R real_sums[256];
    __shared__ R extremes[256];
    __shared__ int indices[256];
    const unsigned lane=threadIdx.x;
    T sum=O::zero();R real_sum=0,extremum=0;int index=0;
    for(std::size_t i=lane;i<std::size_t(n);i+=blockDim.x) {
        const T value=x[i*ix];
        if(op==sb_operation::dot)sum=O::madd(conjugate?O::conj(value):value,y[i*iy],sum);
        else if(op==sb_operation::nrm2)real_sum+=O::norm2(value);
        else if(op==sb_operation::asum)real_sum+=O::abs1(value);
        else {
            const R magnitude=O::abs1(value);
            if(!index||(op==sb_operation::iamax?magnitude>extremum:magnitude<extremum)) {
                extremum=magnitude;index=int(i)+1;
            }
        }
    }
    sums[lane]=sum;real_sums[lane]=real_sum;extremes[lane]=extremum;indices[lane]=index;
    __syncthreads();
    for(unsigned offset=blockDim.x/2;offset;offset/=2) {
        if(lane<offset) {
            sums[lane]=O::add(sums[lane],sums[lane+offset]);
            real_sums[lane]+=real_sums[lane+offset];
            const auto rhs=lane+offset;
            if(indices[rhs]&&(!indices[lane]||
                (op==sb_operation::iamax?extremes[rhs]>extremes[lane]:extremes[rhs]<extremes[lane])||
                (extremes[rhs]==extremes[lane]&&indices[rhs]<indices[lane]))) {
                extremes[lane]=extremes[rhs];indices[lane]=indices[rhs];
            }
        }
        __syncthreads();
    }
    if(lane==0) {
        if(op==sb_operation::dot)output[0]=sums[0];
        else if(op==sb_operation::iamax||op==sb_operation::iamin)*reinterpret_cast<int*>(output)=indices[0];
        else *reinterpret_cast<R*>(output)=op==sb_operation::nrm2?static_cast<R>(sqrt(real_sums[0])):real_sums[0];
    }
}

template<class T> void sb_vector_launch(context& ctx,const std::string& name,int n,T alpha,
        T* x,int ix,T* y,int iy,T* scalar,bool conjugate) {
    const auto op=sb_parse_op(name);
    if(op==sb_operation::dot||op==sb_operation::nrm2||op==sb_operation::asum||op==sb_operation::iamax||op==sb_operation::iamin)
        sb_reduction_kernel<<<1,256,0,ctx.stream>>>(op,n,x,ix,y,iy,scalar,conjugate);
    else sb_vector_kernel<<<std::min<std::size_t>((std::size_t(n)+255)/256,4096),256,0,ctx.stream>>>(op,n,alpha,x,ix,y,iy);
    CUDA_CHECK(cudaGetLastError());
}

template<class T> __global__ void sb_matrix_kernel(sb_operation op,int ta,int tb,bool left,bool lower,bool unit,
        int m,int n,int k,T alpha,const T* a,int lda,const T* b,int ldb,T beta,T* c,int ldc,const T* x,int ix,int iy) {
    using O=sb_ops<T>;
    const int rows=op==sb_operation::gemv?(ta==0?m:n):op==sb_operation::syrk?n:m;
    const int cols=op==sb_operation::gemv?1:n;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<std::size_t(rows)*cols;i+=std::size_t(blockDim.x)*gridDim.x) {
        const int row=int(i%rows),col=int(i/rows);
        if(op==sb_operation::syrk&&((lower&&row<col)||(!lower&&row>col)))continue;
        const auto value=::archbench::software_blas::matrix_element<O>(op,row,col,m,n,k,ta,tb,left,lower,unit,
                    alpha,a,lda,b,ldb,beta,c,ldc,x,ix,iy);
        c[op==sb_operation::gemv?std::size_t(row)*iy:std::size_t(col)*ldc+row]=value;
    }
}

template<class T> __global__ void sb_solve_kernel(int trans,bool left,bool lower,bool unit,
        int m,int n,T alpha,const T* a,int lda,T* c,int ldc) {
    for(int rhs=blockIdx.x*blockDim.x+threadIdx.x;rhs<(left?n:m);rhs+=blockDim.x*gridDim.x)
        ::archbench::software_blas::solve_rhs<sb_ops<T>>(rhs,m,n,trans,left,lower,unit,alpha,a,lda,c,ldc);
}

template<class T> void sb_matrix_launch(context& ctx,const std::string& name,cublasOperation_t ta,cublasOperation_t tb,
        cublasSideMode_t side,cublasFillMode_t uplo,cublasDiagType_t diag,int m,int n,int k,
        T alpha,const T* a,int lda,const T* b,int ldb,T beta,T* c,int ldc,const T* x,int ix,int iy) {
    const auto op=sb_parse_op(name);
    const bool left=side==CUBLAS_SIDE_LEFT,lower=uplo==CUBLAS_FILL_MODE_LOWER,unit=diag==CUBLAS_DIAG_UNIT;
    if(op==sb_operation::syrk&&ta==CUBLAS_OP_C)
        throw std::invalid_argument("SYRK accepts N/T; a Hermitian rank-k operation is a different request");
    if(op==sb_operation::trsm) {
        const int rhs=left?n:m;
        sb_solve_kernel<<<std::min((rhs+127)/128,4096),128,0,ctx.stream>>>(sb_transpose(ta),left,lower,unit,m,n,alpha,a,lda,c,ldc);
    } else {
        const std::size_t count=op==sb_operation::gemv?std::size_t(ta==CUBLAS_OP_N?m:n):std::size_t(op==sb_operation::syrk?n:m)*n;
        sb_matrix_kernel<<<std::min<std::size_t>((count+255)/256,4096),256,0,ctx.stream>>>(op,sb_transpose(ta),sb_transpose(tb),
                    left,lower,unit,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,x,ix,iy);
    }
    CUDA_CHECK(cudaGetLastError());
}
