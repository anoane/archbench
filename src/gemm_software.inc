// SPDX-License-Identifier: AGPL-3.0-or-later
// Included inside archbench::gemm_detail, after the shared input/reference code.
// This is a real CUDA SIMT implementation, not a host timing stand-in.
struct software_gemm_descriptor {
    int m,n,k,batch,lda,ldb,ldc;
    size_t stride_a,stride_b,stride_c;
    kind a,b,c,bias;
    bool trans_a,trans_b,conjugate_a,conjugate_b,add_bias,relu,gelu;
    int block_size;
    gemm_software::input_rounding rounding;
    double2 alpha,beta;
};

template<class T> __device__ inline T software_fma(T a,T b,T c) {
    if constexpr(std::is_same_v<T,float>) return __fmaf_rn(a,b,c);
    else if constexpr(std::is_same_v<T,double>) return __fma_rn(a,b,c);
    else return a*b+c; // unsigned modulo arithmetic; the host guard proves the result is in INT32.
}
template<class T> __device__ inline T software_tanh(T x) {
    if constexpr(std::is_same_v<T,double>) return tanh(x);
    else return T(tanhf(float(x)));
}
template<class T> __device__ inline void software_store(void* destination,kind type,
                                                       size_t index,T re,T im) {
    switch(type) {
    case kind::f16: static_cast<__half*>(destination)[index]=__float2half_rn(float(re));break;
    case kind::bf16: static_cast<__nv_bfloat16*>(destination)[index]=__float2bfloat16_rn(float(re));break;
    case kind::f32: static_cast<float*>(destination)[index]=float(re);break;
    case kind::f64: static_cast<double*>(destination)[index]=double(re);break;
    case kind::c64: static_cast<cuComplex*>(destination)[index]=make_cuComplex(float(re),float(im));break;
    case kind::c128: static_cast<cuDoubleComplex*>(destination)[index]=make_cuDoubleComplex(double(re),double(im));break;
    case kind::i32: static_cast<int32_t*>(destination)[index]=static_cast<int32_t>(re);break;
    default: break; // rejected before launch; no output type is silently substituted.
    }
}

// Each 16x16 block reuses a shared 16-element reduction tile. The outer tile
// loop is grid-strided so all legal batch counts fit without a grid.z limit.
template<class T,bool Complex,bool HalfAccumulation> __global__ void software_gemm_kernel(
        software_gemm_descriptor p,const void* a,const void* b,const void* c,void* d,
        const void* bias,const uint8_t* scale_a,const uint8_t* scale_b) {
    constexpr int tile=16;
    __shared__ T ar[tile][tile],br[tile][tile];
    __shared__ T ai[Complex?tile:1][Complex?tile:1],bi[Complex?tile:1][Complex?tile:1];
    const int tx=threadIdx.x,ty=threadIdx.y;
    const size_t row_tiles=(size_t(p.m)+tile-1)/tile;
    const size_t col_tiles=(size_t(p.n)+tile-1)/tile;
    const size_t tiles_per_batch=row_tiles*col_tiles;
    const size_t total_tiles=tiles_per_batch*size_t(p.batch);
    for(size_t logical_tile=blockIdx.x;logical_tile<total_tiles;logical_tile+=gridDim.x) {
        const int batch=int(logical_tile/tiles_per_batch);
        const size_t within=logical_tile%tiles_per_batch;
        const int row=int(within%row_tiles)*tile+tx;
        const int col=int(within/row_tiles)*tile+ty;
        T sum_real=0,sum_imag=0;
        for(size_t reduction=0;reduction<size_t(p.k);reduction+=tile) {
            const size_t ka=reduction+ty,kb=reduction+tx;
            T av=0,bv=0,avi=0,bvi=0;
            if(row<p.m&&ka<size_t(p.k)) {
                const size_t index=gemm_software::matrix_index(row,int(ka),batch,p.lda,p.stride_a,p.trans_a);
                av=gemm_software::load_component<T>(a,p.a,index);
                if constexpr(Complex) {
                    avi=gemm_software::load_component<T>(a,p.a,index,true);
                    if(p.conjugate_a)avi=-avi;
                }
                if constexpr(std::is_same_v<T,float>) av=gemm_software::round_input(av,p.rounding);
                if constexpr(!std::is_integral_v<T>) {
                    const T scale=T(gemm_software::load_scale(scale_a,row,int(ka),p.k,p.block_size));
                    av*=scale;avi*=scale;
                }
            }
            if(col<p.n&&kb<size_t(p.k)) {
                const size_t index=gemm_software::matrix_index(int(kb),col,batch,p.ldb,p.stride_b,p.trans_b);
                bv=gemm_software::load_component<T>(b,p.b,index);
                if constexpr(Complex) {
                    bvi=gemm_software::load_component<T>(b,p.b,index,true);
                    if(p.conjugate_b)bvi=-bvi;
                }
                if constexpr(std::is_same_v<T,float>) bv=gemm_software::round_input(bv,p.rounding);
                if constexpr(!std::is_integral_v<T>) {
                    const T scale=T(gemm_software::load_scale(scale_b,col,int(kb),p.k,p.block_size));
                    bv*=scale;bvi*=scale;
                }
            }
            ar[ty][tx]=av;br[ty][tx]=bv;
            if constexpr(Complex) {ai[ty][tx]=avi;bi[ty][tx]=bvi;}
            __syncthreads();
            const int live=int(size_t(p.k)-reduction<tile?size_t(p.k)-reduction:tile);
            for(int q=0;q<live;++q) {
                if constexpr(HalfAccumulation)
                    sum_real=__half2float(__hfma(__float2half_rn(float(ar[q][tx])),
                        __float2half_rn(float(br[ty][q])),__float2half_rn(float(sum_real))));
                else {
                    sum_real=software_fma(ar[q][tx],br[ty][q],sum_real);
                    if constexpr(Complex) {
                        sum_real=software_fma(-ai[q][tx],bi[ty][q],sum_real);
                        sum_imag=software_fma(ar[q][tx],bi[ty][q],sum_imag);
                        sum_imag=software_fma(ai[q][tx],br[ty][q],sum_imag);
                    }
                }
            }
            __syncthreads();
        }
        if(row<p.m&&col<p.n) {
            const size_t ci=gemm_software::matrix_index(row,col,batch,p.ldc,p.stride_c,false);
            const T cr=gemm_software::load_component<T>(c,p.c,ci);
            T answer=0,imag=0;
            if constexpr(std::is_integral_v<T>) {
                const T alpha=static_cast<T>(static_cast<int32_t>(p.alpha.x));
                const T beta=static_cast<T>(static_cast<int32_t>(p.beta.x));
                answer=alpha*sum_real+beta*cr;
            } else if constexpr(HalfAccumulation) {
                answer=__half2float(__hadd(__hmul(__float2half_rn(float(p.alpha.x)),__float2half_rn(float(sum_real))),
                    __hmul(__float2half_rn(float(p.beta.x)),__float2half_rn(float(cr)))));
            } else {
                answer=software_fma(T(p.alpha.x),sum_real,T(p.beta.x)*cr);
                if constexpr(Complex) {
                    const T ci_imag=gemm_software::load_component<T>(c,p.c,ci,true);
                    answer=software_fma(-T(p.alpha.y),sum_imag,answer);
                    answer=software_fma(-T(p.beta.y),ci_imag,answer);
                    imag=software_fma(T(p.alpha.x),sum_imag,T(p.alpha.y)*sum_real);
                    imag=software_fma(T(p.beta.x),ci_imag,imag);
                    imag=software_fma(T(p.beta.y),cr,imag);
                }
            }
            if(p.add_bias) {
                const T bias_real=gemm_software::load_component<T>(bias,p.bias,row);
                if constexpr(HalfAccumulation)
                    answer=__half2float(__hadd(__float2half_rn(float(answer)),__float2half_rn(float(bias_real))));
                else answer+=bias_real;
                if constexpr(Complex)imag+=gemm_software::load_component<T>(bias,p.bias,row,true);
            }
            if(p.relu) {
                if constexpr(std::is_integral_v<T>) {if(static_cast<int32_t>(answer)<0)answer=0;}
                else if(answer<T(0))answer=0;
            }
            if(p.gelu) {
                if constexpr(!std::is_integral_v<T>)
                    answer=T(.5)*answer*(T(1)+software_tanh(T(.79788456080286535588)*
                        (answer+T(.044715)*answer*answer*answer)));
            }
            software_store(d,p.c,ci,answer,imag);
        }
        __syncthreads();
    }
}

inline void software_gemm(context& ctx,result& out) {
    const auto start=clock_type::now();
    const problem p(ctx.opts);
    const std::string dtype=ctx.opts.get("dtype","f32");
    const auto a=parse_type(dtype),b=parse_type(ctx.opts.get("dtype-b",dtype));
    const auto c=parse_type(ctx.opts.get("output-dtype",a.narrow?"f16":dtype=="i8"?"i32":dtype));
    const auto requested_compute=ctx.opts.get("compute","default");
    // Parse the public compute vocabulary even for a mathematical reference.
    const auto requested_type=compute_type(ctx.opts,a);
    const auto policy=ctx.opts.get("software-policy","same-request");
    if(policy!="same-request"&&policy!="mathematical-reference")
        throw std::invalid_argument("software-policy must be same-request or mathematical-reference");
    const bool mathematical_reference=policy=="mathematical-reference";
    out.metadata["backend"]="software";
    out.metrics["requested_cublas_algorithm_enum"]=static_cast<int>(classic_gemm_algorithm(ctx.opts));
    out.metadata["cublas_algorithm_control"]=ctx.opts.get("cublas-algo","default");
    out.metadata["cublas_algorithm_control_application"]="source-request provenance only; custom SIMT kernel does not call GemmEx";
    out.metadata["requested_compute_policy"]=requested_compute;
    out.metadata["software_policy"]=policy;
    out.metadata["policy_equivalence"]=mathematical_reference?"false":"true";
    out.metadata["policy_equivalence_scope"]="stored values and declared compute format; no bitwise, reduction-order, library-algorithm, or instruction equivalence";
    if(!mathematical_reference&&(requested_compute=="fixedpoint"||requested_compute=="bf16x9"||ctx.opts.flag("fast-accum",false))) {
        skip(out,"software same-request does not emulate the library fixed-point/BF16x9/fast-accum algorithm; use --software-policy mathematical-reference for an explicitly nonequivalent numerical baseline");
        out.metadata["policy_equivalence"]="false";return;
    }
    if(c.narrow||c.id==kind::i8) {
        skip(out,"software GEMM implements f16/bf16/f32/f64/c64/c128/i32 outputs; narrow output quantization and i8 output are not defined by this module");return;
    }
    const auto epi=epilogue(ctx.opts.get("epilogue","default"));
    const bool integer=a.id==kind::i8||a.id==kind::i32;
    const bool other_integer=b.id==kind::i8||b.id==kind::i32;
    const bool complex=a.complex||b.complex||c.complex;
    const bool double_precision=a.id==kind::f64||b.id==kind::f64||c.id==kind::f64||
        a.id==kind::c128||b.id==kind::c128||c.id==kind::c128;
    if(integer!=other_integer||(integer&&(c.id!=kind::i32||complex))||(!integer&&c.id==kind::i32)) {
        skip(out,"software GEMM requires both integer operands with i32 output, or both floating operands with floating output");return;
    }
    if(complex&&(!(a.complex&&b.complex&&c.complex)||
       (double_precision&&(a.id!=kind::c128||b.id!=kind::c128||c.id!=kind::c128)))) {
        skip(out,"software complex GEMM requires c64/c64/c64 or c128/c128/c128 storage");return;
    }
    if((complex&&(epi.relu||epi.gelu))||(integer&&epi.gelu)) {
        skip(out,"software GELU requires real floating output; ReLU is not defined for complex output");return;
    }
    if(double_precision&&!complex&&c.id!=kind::f64) {
        skip(out,"software real FP64 GEMM requires f64 output; no implicit precision substitution");return;
    }
    const bool half_accumulation=!mathematical_reference&&(requested_compute=="f16"||requested_compute=="f16_pedantic");
    if(half_accumulation&&(a.id!=kind::f16||b.id!=kind::f16||c.id!=kind::f16)) {
        skip(out,"software half accumulation requires f16 A, B, and output");return;
    }
    auto rounding=gemm_software::input_rounding::unchanged;
    if(!mathematical_reference) {
        if(requested_compute=="tf32")rounding=gemm_software::input_rounding::tf32;
        if(requested_compute=="fast_f16")rounding=gemm_software::input_rounding::f16;
        if(requested_compute=="fast_bf16")rounding=gemm_software::input_rounding::bf16;
        if(rounding!=gemm_software::input_rounding::unchanged&&
           (a.id!=kind::f32||b.id!=kind::f32||c.id!=kind::f32)) {
            skip(out,"software down-conversion policies require f32 A, B, and output");return;
        }
        if((integer&&requested_compute!="default"&&requested_compute!="pedantic")||
           (double_precision&&requested_compute!="default"&&requested_compute!="pedantic")) {
            skip(out,"requested compute policy is incompatible with the software integer/FP64 storage path");return;
        }
        if(double_precision&&requested_type!=CUBLAS_COMPUTE_64F&&requested_type!=CUBLAS_COMPUTE_64F_PEDANTIC) {
            skip(out,"the requested mixed storage infers FP32 compute from A but requires FP64 output; use mathematical-reference for a widest-precision numerical baseline");return;
        }
    }
    const auto compute=mathematical_reference?
        (integer?CUBLAS_COMPUTE_32I:double_precision?CUBLAS_COMPUTE_64F:CUBLAS_COMPUTE_32F):requested_type;
    // Even a mathematical reference preserves alpha/beta's requested scalar
    // storage rounding; only the implementation of the dot product changes.
    scalar_pair scalar(ctx.opts,c,mathematical_reference?requested_type:compute);
    if(!std::isfinite(static_cast<double>(scalar.alpha.real()))||!std::isfinite(static_cast<double>(scalar.beta.real()))||
       !std::isfinite(static_cast<double>(scalar.alpha.imag()))||!std::isfinite(static_cast<double>(scalar.beta.imag())))
        throw std::invalid_argument("alpha/beta overflow the selected scalar storage type");
    if(integer&&(9.L*p.k*std::abs(scalar.alpha)+3.L*std::abs(scalar.beta)+(epi.bias?3:0)>INT32_MAX))
        throw std::invalid_argument("integer reference would exceed INT32; reduce K or alpha/beta");
    const bool fp4=a.id==kind::e2m1||b.id==kind::e2m1;
    const auto scale_mode=ctx.opts.get("scale-mode",fp4?"block16":"tensor");
    const int block_size=scale_mode=="block16"?16:scale_mode=="block32"?32:0;
    if(scale_mode!="tensor"&&scale_mode!="block16"&&scale_mode!="block32")
        throw std::invalid_argument("scale-mode must be tensor, block16, or block32");
    if(block_size&&(!a.narrow||!b.narrow)) {
        skip(out,"software block scaling requires narrow A and B inputs");return;
    }
    if((block_size==16&&(!fp4||a.id!=kind::e2m1||b.id!=kind::e2m1))||(block_size==32&&fp4)) {
        skip(out,"software block16 matches FP4 E2M1/UE4M3 and block32 matches FP8/UE8M0; mixed scale formats are not defined");return;
    }
    const float scale_a=float(ctx.opts.real("scale-a",1)),scale_b=float(ctx.opts.real("scale-b",1));
    if(!std::isfinite(scale_a)||!std::isfinite(scale_b)||scale_a<=0||scale_b<=0)
        throw std::invalid_argument("tensor scales must be finite and positive");
    if(block_size&&(scale_a!=1||scale_b!=1))
        throw std::invalid_argument("block scale arrays are generated independently; scale-a/scale-b apply only to tensor scaling");
    if(!(a.narrow||b.narrow)&&(scale_a!=1||scale_b!=1))
        throw std::invalid_argument("A/B scales apply to narrow-input matmul only");
    const auto batch_mode=ctx.opts.get("batch-mode","strided");
    if(batch_mode!="strided"&&batch_mode!="pointer")throw std::invalid_argument("batch-mode must be strided or pointer");
    const auto bias_type=a.narrow&&c.id==kind::f32?parse_type("bf16"):c;
    const size_t ab=storage_bytes(p.count_a,a),bb=storage_bytes(p.count_b,b),cb=storage_bytes(p.count_c,c);
    const size_t bias_bytes=epi.bias?storage_bytes(p.m,bias_type):0;
    const size_t sa_count=block_size?block_scale_count(p.m,p.k,block_size):sizeof(float);
    const size_t sb_count=block_size?block_scale_count(p.n,p.k,block_size):sizeof(float);
    ctx.memory_guard(sum_sizes({ab,bb,product(2,cb),bias_bytes,sa_count,sb_count}));
    device_buffer<uint8_t> da(ab),db(bb),dc(cb),dd(cb),bias(bias_bytes),sa(sa_count),sb(sb_count);
    fill_gemm_inputs(ctx,p,a,b,c,da.get(),db.get(),dc.get());
    if(epi.bias)fill_buffer(ctx,bias.get(),bias_type,p.m,97);
    if(block_size) {
        const auto hsa=make_block_scales(sa_count,41,block_size==16),hsb=make_block_scales(sb_count,53,block_size==16);
        CUDA_CHECK(cudaMemcpyAsync(sa.get(),hsa.data(),sa_count,cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(sb.get(),hsb.data(),sb_count,cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(sa.get(),&scale_a,sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(sb.get(),&scale_b,sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
    }
    const software_gemm_descriptor descriptor{p.m,p.n,p.k,p.batch,p.lda,p.ldb,p.ldc,
        p.stride_a,p.stride_b,p.stride_c,a.id,b.id,c.id,bias_type.id,
        p.ta!=CUBLAS_OP_N,p.tb!=CUBLAS_OP_N,p.ta==CUBLAS_OP_C,p.tb==CUBLAS_OP_C,
        epi.bias,epi.relu,epi.gelu,block_size,rounding,
        make_double2(double(scalar.alpha.real()),double(scalar.alpha.imag())),
        make_double2(double(scalar.beta.real()),double(scalar.beta.imag()))};
    const size_t tiles=product(product((size_t(p.m)+15)/16,(size_t(p.n)+15)/16),p.batch);
    const int blocks=int(std::min<size_t>(tiles,4096));
    const auto launch=[&] {
#define SOFTWARE_GEMM_LAUNCH(T,C,H) software_gemm_kernel<T,C,H><<<blocks,dim3(16,16),0,ctx.stream>>>(descriptor,da.get(),db.get(),dc.get(),dd.get(),bias.get(),sa.get(),sb.get())
        if(integer) {SOFTWARE_GEMM_LAUNCH(uint32_t,false,false);}
        else if(complex&&double_precision) {SOFTWARE_GEMM_LAUNCH(double,true,false);}
        else if(complex) {SOFTWARE_GEMM_LAUNCH(float,true,false);}
        else if(double_precision) {SOFTWARE_GEMM_LAUNCH(double,false,false);}
        else if(half_accumulation) {SOFTWARE_GEMM_LAUNCH(float,false,true);}
        else {SOFTWARE_GEMM_LAUNCH(float,false,false);}
#undef SOFTWARE_GEMM_LAUNCH
        CUDA_CHECK(cudaGetLastError());
    };
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    out.metrics["setup_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-start).count();
    record_problem(out,p,a,b,c,std::abs(scalar.beta)!=0);
    out.execution="software_simt_gemm";
    out.metadata["execution_evidence"]="custom CUDA kernel; explicit SIMT arithmetic and storage decoding; hardware instructions still require SASS evidence";
    out.metadata["implementation_policy"]=integer?"int32 exact bounded mathematical result":
        half_accumulation?"scalar half FMA with half accumulator":
        double_precision?"FP64 SIMT FMA":rounding==gemm_software::input_rounding::tf32?"RNE TF32 operands; FP32 SIMT FMA":
        rounding==gemm_software::input_rounding::f16?"RNE binary16 operands; FP32 SIMT FMA":
        rounding==gemm_software::input_rounding::bf16?"RNE bfloat16 operands; FP32 SIMT FMA":"FP32 SIMT FMA";
    out.metadata["software_reduction_order"]="ascending K; one output per thread; fixed 16x16 shared tiles";
    out.metadata["software_library_controls"]="workspace, heuristic, algorithm tuning, atomics allowance and SM target are library scheduling controls; not applied to custom kernel";
    out.metadata["batch_mode_requested"]=batch_mode;
    out.metadata["batch_addressing"]="mathematically identical regular batch strides; no cuBLAS pointer-array API overhead";
    out.metadata["scale_mode"]=scale_mode;
    if(block_size)out.metadata["block_scale_batch_semantics"]="same tiled row/column scale arrays reused for each batch";
    out.metadata["timing"]="cuda_event_batch; separate immutable C and D";
    out.metadata["epilogue_arithmetic"]=half_accumulation?"half alpha/beta and bias, float nonlinear activation, final output conversion":
        "accumulator-precision alpha/beta, bias and activation; single final storage conversion";
    out.metrics["scale_storage_bytes"]=double(sa_count+sb_count);
    out.metrics["software_tile_m"]=16;out.metrics["software_tile_n"]=16;out.metrics["software_tile_k"]=16;
    launch();CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    validate_gemm(ctx,out,p,a,b,c,bias_type,scalar,dd.get(),epi,block_size,scale_a,scale_b,mathematical_reference?"default":"");
    if(out.validation!="passed")return;
    out.samples_ms=ctx.measure(launch);
    validate_gemm(ctx,out,p,a,b,c,bias_type,scalar,dd.get(),epi,block_size,scale_a,scale_b,mathematical_reference?"default":"");
    out.metadata["validation_timing_phase"]="before and after performance measurement";
}
