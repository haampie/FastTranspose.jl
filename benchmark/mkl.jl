using LinearAlgebra, FastTranspose

# void mkl_domatcopy(char ordering, char trans, size_t rows, size_t cols, const double alpha, const double * A, size_t lda, double * B, size_t ldb);

function mkl_matcopy!(B::Matrix{Float64}, A::Matrix{Float64}, alpha = 1.0)
    m, n = size(A)
    ordering = 'C'
    trans = 'T'
    ccall((:mkl_domatcopy, BLAS.libblas), Cvoid, (Ref{Cchar}, Ref{Cchar}, Ref{Csize_t}, Ref{Csize_t}, Ref{Cdouble}, Ptr{Float64}, Ref{Csize_t}, Ptr{Float64}, Ref{Csize_t}), 
           ordering, trans, m, n, alpha, A, m, B, n)

    return nothing
end

function mkl_matcopy!(B::Matrix{Float32}, A::Matrix{Float32}, alpha = 1.0)
    m, n = size(A)
    ordering = 'C'
    trans = 'T'
    ccall((:mkl_somatcopy, BLAS.libblas), Cvoid, (Ref{Cchar}, Ref{Cchar}, Ref{Csize_t}, Ref{Csize_t}, Ref{Cdouble}, Ptr{Float64}, Ref{Csize_t}, Ptr{Float64}, Ref{Csize_t}), 
           ordering, trans, m, n, alpha, A, m, B, n)

    return nothing
end

using BenchmarkTools

function bench(m = 4096, n = m)
    BLAS.set_num_threads(1)

    A = rand(m, n)
    B = rand(n, m)

    mkl = @benchmark mkl_matcopy!($B, $A)
    me = @benchmark recursive_transpose!($B, $A)

    return mkl, me
end
