using Test, LinearAlgebra
using FastTranspose

A = rand(123, 1423)
B = rand(1423, 123)

@assert norm(A' - B) != 0

recursive_transpose!(B, A)

@test(norm(A' - B) == 0)