# FastTranspose.jl

Experiment to transpose matrices of `Float64` out of place really fast for CPUs that support AVX2. Probably something similar can be done for `Float32`.

It's a recursive cache-oblivious algorithm, with a kernel that exploits AVX2.

From the example below, it seems 3.9x slower than a memcpy, and 4.5x faster than julia base.

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