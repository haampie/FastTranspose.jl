module FastTranspose

export recursive_transpose!

using SIMDPirates

const Vec64_2 = SVec{2,Float64}
const Vec64_4 = SVec{4,Float64}

canonicalize(A) = A
canonicalize(A::Union{<:SubArray}) = canonicalize(parent(A))

function kernel64!(B::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, scaling)

    m, n = size(A)

    Ac = canonicalize(A)
    Bc = canonicalize(B)

    sA = stride(Ac, 2)
    sB = stride(Bc, 2)

    @inbounds for i = Base.OneTo(m ÷ 4)
        pA = pointer(A) + 4(i - 1) * sizeof(Float64)
        pB = pointer(B) + 4(i - 1) * sB * sizeof(Float64)

        for j = Base.OneTo(n ÷ 4)
            
            v00 = vload(SVec{4,Float64}, pA + 0 * sA * sizeof(Float64))
            v01 = vload(SVec{4,Float64}, pA + 1 * sA * sizeof(Float64))
            v02 = vload(SVec{4,Float64}, pA + 2 * sA * sizeof(Float64))
            v03 = vload(SVec{4,Float64}, pA + 3 * sA * sizeof(Float64))

            v04 = SIMDPirates.shufflevector(v00, v01, Val{(0, 4, 2, 6)}())
            v05 = SIMDPirates.shufflevector(v00, v01, Val{(1, 5, 3, 7)}())
            v06 = SIMDPirates.shufflevector(v02, v03, Val{(0, 4, 2, 6)}())
            v07 = SIMDPirates.shufflevector(v02, v03, Val{(1, 5, 3, 7)}())
            v08 = SIMDPirates.shufflevector(v04, v06, Val{(0, 1, 4, 5)}())
            v09 = SIMDPirates.shufflevector(v05, v07, Val{(0, 1, 4, 5)}())
            v10 = SIMDPirates.shufflevector(v04, v06, Val{(2, 3, 6, 7)}())
            v11 = SIMDPirates.shufflevector(v05, v07, Val{(2, 3, 6, 7)}())

            v12 = scaling * v08
            v13 = scaling * v09
            v14 = scaling * v10
            v15 = scaling * v11

            vstore!(pB + 0 * sB * sizeof(Float64), v12)
            vstore!(pB + 1 * sB * sizeof(Float64), v13)
            vstore!(pB + 2 * sB * sizeof(Float64), v14)
            vstore!(pB + 3 * sB * sizeof(Float64), v15)

            pA += 4 * sA * sizeof(Float64)
            pB += 4 * sizeof(Float64)
        end
    end

    return B
end

# Todo, clean up ... the cleanup business
function kernel64_with_cleanup!(B::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, scaling)

    m, n = size(A)

    Ac = canonicalize(A)
    Bc = canonicalize(B)

    sA = stride(Ac, 2)
    sB = stride(Bc, 2)

    i = 1
    @inbounds begin
        while i + 4 ≤ m
            pA = pointer(A) + (i - 1) * sizeof(Float64)
            pB = pointer(B) + (i - 1) * sB * sizeof(Float64)

            j = 1
            while j + 4 ≤ n
                v00 = vload(SVec{4,Float64}, pA + 0 * sA * sizeof(Float64))
                v01 = vload(SVec{4,Float64}, pA + 1 * sA * sizeof(Float64))
                v02 = vload(SVec{4,Float64}, pA + 2 * sA * sizeof(Float64))
                v03 = vload(SVec{4,Float64}, pA + 3 * sA * sizeof(Float64))
    
                v04 = SIMDPirates.shufflevector(v00, v01, Val{(0, 4, 2, 6)}())
                v05 = SIMDPirates.shufflevector(v00, v01, Val{(1, 5, 3, 7)}())
                v06 = SIMDPirates.shufflevector(v02, v03, Val{(0, 4, 2, 6)}())
                v07 = SIMDPirates.shufflevector(v02, v03, Val{(1, 5, 3, 7)}())
                v08 = SIMDPirates.shufflevector(v04, v06, Val{(0, 1, 4, 5)}())
                v09 = SIMDPirates.shufflevector(v05, v07, Val{(0, 1, 4, 5)}())
                v10 = SIMDPirates.shufflevector(v04, v06, Val{(2, 3, 6, 7)}())
                v11 = SIMDPirates.shufflevector(v05, v07, Val{(2, 3, 6, 7)}())
    
                v12 = scaling * v08
                v13 = scaling * v09
                v14 = scaling * v10
                v15 = scaling * v11
    
                vstore!(pB + 0 * sB * sizeof(Float64), v12)
                vstore!(pB + 1 * sB * sizeof(Float64), v13)
                vstore!(pB + 2 * sB * sizeof(Float64), v14)
                vstore!(pB + 3 * sB * sizeof(Float64), v15)

                pA += 4 * sA * sizeof(Float64)
                pB += 4 * sizeof(Float64)
                j += 4
            end

            while j ≤ n
                B[j, i + 0] = scaling * A[i + 0, j]
                B[j, i + 1] = scaling * A[i + 1, j]
                B[j, i + 2] = scaling * A[i + 2, j]
                B[j, i + 3] = scaling * A[i + 3, j]

                j += 1
            end

            i += 4
        end

        while i ≤ m
            pA = pointer(A) + (i - 1) * sizeof(Float64)
            pB = pointer(B) + (i - 1) * sB * sizeof(Float64)

            j = 1
            while j + 4 ≤ n
                B[j + 0, i] = scaling * A[i, j + 0]
                B[j + 1, i] = scaling * A[i, j + 1]
                B[j + 2, i] = scaling * A[i, j + 2]
                B[j + 3, i] = scaling * A[i, j + 3]

                j += 4
            end

            while j ≤ n
                B[j, i] = scaling * A[i, j]
                j += 1
            end

            i += 1
        end
    end

    return B
end

function transpose_avx_multiples_of_eight!(B::AbstractMatrix{Float32}, A::AbstractMatrix{Float32})

    m, n = size(A)

    Ac = canonicalize(A)
    Bc = canonicalize(B)

    sA = stride(Ac, 2)
    sB = stride(Bc, 2)

    @inbounds for i = Base.OneTo(m ÷ 8)
        pA = pointer(A) + 8(i - 1) * sizeof(Float32)
        pB = pointer(B) + 8(i - 1) * sB * sizeof(Float32)

        for j = Base.OneTo(n ÷ 8)
            
            v1 = vload(SVec{8,Float32}, pA + 0 * sA * sizeof(Float32))
            v2 = vload(SVec{8,Float32}, pA + 1 * sA * sizeof(Float32))
            v3 = vload(SVec{8,Float32}, pA + 2 * sA * sizeof(Float32))
            v4 = vload(SVec{8,Float32}, pA + 3 * sA * sizeof(Float32))
            v5 = vload(SVec{8,Float32}, pA + 4 * sA * sizeof(Float32))
            v6 = vload(SVec{8,Float32}, pA + 5 * sA * sizeof(Float32))
            v7 = vload(SVec{8,Float32}, pA + 6 * sA * sizeof(Float32))
            v8 = vload(SVec{8,Float32}, pA + 7 * sA * sizeof(Float32))

            w1 = SVec{8,Float32}(v1[1], v2[1], v3[1], v4[1], v5[1], v6[1], v7[1], v8[1])
            w2 = SVec{8,Float32}(v1[2], v2[2], v3[2], v4[2], v5[2], v6[2], v7[2], v8[2])
            w3 = SVec{8,Float32}(v1[3], v2[3], v3[3], v4[3], v5[3], v6[3], v7[3], v8[3])
            w4 = SVec{8,Float32}(v1[4], v2[4], v3[4], v4[4], v5[4], v6[4], v7[4], v8[4])
            w5 = SVec{8,Float32}(v1[5], v2[5], v3[5], v4[5], v5[5], v6[5], v7[5], v8[5])
            w6 = SVec{8,Float32}(v1[6], v2[6], v3[6], v4[6], v5[6], v6[6], v7[6], v8[6])
            w7 = SVec{8,Float32}(v1[7], v2[7], v3[7], v4[7], v5[7], v6[7], v7[7], v8[7])
            w8 = SVec{8,Float32}(v1[8], v2[8], v3[8], v4[8], v5[8], v6[8], v7[8], v8[8])

            vstore!(pB + 0 * sB * sizeof(Float32), w1)
            vstore!(pB + 1 * sB * sizeof(Float32), w2)
            vstore!(pB + 2 * sB * sizeof(Float32), w3)
            vstore!(pB + 3 * sB * sizeof(Float32), w4)
            vstore!(pB + 4 * sB * sizeof(Float32), w5)
            vstore!(pB + 5 * sB * sizeof(Float32), w6)
            vstore!(pB + 6 * sB * sizeof(Float32), w7)
            vstore!(pB + 7 * sB * sizeof(Float32), w8)

            pA += 8 * sA * sizeof(Float32)
            pB += 8 * sizeof(Float32)
        end
    end

    return B
end

power_of_two_smaller_than_or_equal_to(k::T) where {T} = (1 << (8sizeof(T) - leading_zeros(k) - 1)) % Int

function recursive_transpose!(A::AbstractMatrix{T}, B::AbstractMatrix, scaling = one(T), b::Val{blocksize} = Val(32)) where {T,blocksize}
    m, n = size(A)

    if max(m, n) ≤ blocksize
        if rem(m, 4) == rem(n, 4) == 0
            kernel64!(A, B, scaling)
        else
            kernel64_with_cleanup!(A, B, scaling)
        end
    else
        k = power_of_two_smaller_than_or_equal_to(max(m, n) ÷ 2)
        if m > n
            recursive_transpose!(view(A, 1:k, :), view(B, :, 1:k), scaling, b)
            recursive_transpose!(view(A, k+1:m, :), view(B, :, k+1:m), scaling, b)
        else
            recursive_transpose!(view(A, :, 1:k), view(B, 1:k, :), scaling, b)
            recursive_transpose!(view(A, :, k+1:n), view(B, k+1:n, :), scaling, b)
        end
    end

    return nothing
end

function recursive_transpose!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, b::Val{blocksize} = Val(32)) where blocksize
    @assert rem(size(B, 1), 8) == 0
    @assert rem(size(B, 2), 8) == 0
    recursive_transpose_impl!(A, B, b)
end

function recursive_transpose_impl!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, b::Val{blocksize}) where blocksize
    m, n = size(A)

    if max(m, n) ≤ blocksize
        transpose_avx_multiples_of_eight!(A, B)
    else
        k = power_of_two_smaller_than_or_equal_to(max(m, n) ÷ 2)
        if m > n
            recursive_transpose_impl!(view(A, 1:k, :), view(B, :, 1:k), b)
            recursive_transpose_impl!(view(A, k+1:m, :), view(B, :, k+1:m), b)
        else
            recursive_transpose_impl!(view(A, :, 1:k), view(B, 1:k, :), b)
            recursive_transpose_impl!(view(A, :, k+1:n), view(B, k+1:n, :), b)
        end
    end

    return nothing
end



# function fast_transpose_sse!(B::Matrix{Float64}, A::Matrix{Float64})

#     m, n = size(A)

#     sA = stride(A, 2)
#     sB = stride(B, 2)

#     @inbounds for i = Base.OneTo(m ÷ 4)
#         pA = pointer(A) + 4(i - 1) * sizeof(Float64)
#         pB = pointer(B) + 4(i - 1) * sB * sizeof(Float64)

#         for j = Base.OneTo(n ÷ 4)
            
#             v1 = vload(Vec64_2, pA + 0 * sA * sizeof(Float64))
#             v2 = vload(Vec64_2, pA + 1 * sA * sizeof(Float64))
#             v3 = vload(Vec64_2, pA + 2 * sA * sizeof(Float64))
#             v4 = vload(Vec64_2, pA + 3 * sA * sizeof(Float64))

#             v5 = vload(Vec64_2, pA + 0 * sA * sizeof(Float64) + 2 * sizeof(Float64))
#             v6 = vload(Vec64_2, pA + 1 * sA * sizeof(Float64) + 2 * sizeof(Float64))
#             v7 = vload(Vec64_2, pA + 2 * sA * sizeof(Float64) + 2 * sizeof(Float64))
#             v8 = vload(Vec64_2, pA + 3 * sA * sizeof(Float64) + 2 * sizeof(Float64))

#             w1 = Vec64_2(v1[1], v2[1])
#             w2 = Vec64_2(v1[2], v2[2])
#             w3 = Vec64_2(v3[1], v4[1])
#             w4 = Vec64_2(v3[2], v4[2])

#             w5 = Vec64_2(v5[1], v6[1])
#             w6 = Vec64_2(v5[2], v6[2])
#             w7 = Vec64_2(v7[1], v8[1])
#             w8 = Vec64_2(v7[2], v8[2])

#             vstore!(pB + 0 * sB * sizeof(Float64), w1)
#             vstore!(pB + 0 * sB * sizeof(Float64) + 2 * sizeof(Float64), w3)

#             vstore!(pB + 1 * sB * sizeof(Float64), w2)
#             vstore!(pB + 1 * sB * sizeof(Float64) + 2 * sizeof(Float64), w4)

#             vstore!(pB + 2 * sB * sizeof(Float64), w5)
#             vstore!(pB + 2 * sB * sizeof(Float64) + 2 * sizeof(Float64), w7)

#             vstore!(pB + 3 * sB * sizeof(Float64), w6)
#             vstore!(pB + 3 * sB * sizeof(Float64) + 2 * sizeof(Float64), w8)

#             pA += 4 * sA * sizeof(Float64)
#             pB += 4 * sizeof(Float64)
#         end
#     end

#     return B
# end

end