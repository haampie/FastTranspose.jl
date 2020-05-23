[![Build Status](https://travis-ci.com/haampie/FastTranspose.jl.svg?branch=master)](https://travis-ci.com/haampie/FastTranspose.jl)

# FastTranspose.jl

Experiment to transpose matrices of `Float32` and `Float64` out of place really fast for CPUs that support AVX2.

It's a recursive cache-oblivious algorithm, with a kernel that exploits AVX2.

Currently single-threaded, outperforms single-threaded MKL (it does not scale to multiple threads as it's memory-bound).

Install using `] add https://github.com/haampie/FastTranspose.jl.git` in the REPL.

## Example 1: Matrices of size 4096 x 4096

For matrices of order 4096 FastTranpose is 2.8x slower than memcpy and 9.9x faster than Julia's implementation.

```julia
using FastTranpose, BenchmarkTools

julia> B = rand(4096, 4096); A = rand(4096, 4096);

julia> @benchmark recursive_transpose!($B, $A)
BenchmarkTools.Trial: 
  memory estimate:  512 bytes
  allocs estimate:  8
  --------------
  minimum time:     20.838 ms (0.00% GC)
  median time:      21.051 ms (0.00% GC)
  mean time:        21.061 ms (0.00% GC)
  maximum time:     21.452 ms (0.00% GC)
  --------------
  samples:          238
  evals/sample:     1

julia> @benchmark copyto!($B, $A)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     7.243 ms (0.00% GC)
  median time:      7.336 ms (0.00% GC)
  mean time:        7.336 ms (0.00% GC)
  maximum time:     7.891 ms (0.00% GC)
  --------------
  samples:          682
  evals/sample:     1

julia> @benchmark copyto!($B, $(A'))
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     206.161 ms (0.00% GC)
  median time:      206.261 ms (0.00% GC)
  mean time:        206.392 ms (0.00% GC)
  maximum time:     208.590 ms (0.00% GC)
  --------------
  samples:          25
  evals/sample:     1
```

In this case FastTranspose.jl is roughly 1.7x faster than MKL (both single-threaded):

```julia
julia> include("path/to/FastTranspose.jl/benchmark/mkl.jl")

julia> julia> bench(4096)
(Trial(36.533 ms), Trial(20.895 ms)) # (mkl, ours)
```

## Example 2: Very large matrices

For matrices of size `12234 x 13455`, it seems 3.9x slower than a memcpy, and 4.5x faster than Julia's implementation.

```julia
julia> A = rand(12234, 13455); B = similar(A');

julia> @benchmark copyto!($(vec(B)), $(vec(A)))
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     67.904 ms (0.00% GC)
  median time:      69.077 ms (0.00% GC)
  mean time:        69.036 ms (0.00% GC)
  maximum time:     70.404 ms (0.00% GC)
  --------------
  samples:          73
  evals/sample:     1

julia> @benchmark recursive_transpose!($B, $A)
BenchmarkTools.Trial: 
  memory estimate:  512 bytes
  allocs estimate:  8
  --------------
  minimum time:     263.363 ms (0.00% GC)
  median time:      264.583 ms (0.00% GC)
  mean time:        265.051 ms (0.00% GC)
  maximum time:     267.164 ms (0.00% GC)
  --------------
  samples:          19
  evals/sample:     1

julia> @benchmark copyto!($B, $(A'))
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.186 s (0.00% GC)
  median time:      1.207 s (0.00% GC)
  mean time:        1.203 s (0.00% GC)
  maximum time:     1.211 s (0.00% GC)
  --------------
  samples:          5
  evals/sample:     1
```
