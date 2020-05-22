module FastTranspose

export recursive_transpose!

using SIMDPirates

const Vec64_2 = SVec{2,Float64}
const Vec64_4 = SVec{4,Float64}

canonicalize(A) = A
canonicalize(A::Union{<:SubArray}) = canonicalize(parent(A))

function transpose_avx_multiples_of_four!(B::AbstractMatrix{Float64}, A::AbstractMatrix{Float64})

    m, n = size(A)

    Ac = canonicalize(A)
    Bc = canonicalize(B)

    sA = stride(Ac, 2)
    sB = stride(Bc, 2)

    @inbounds for i = Base.OneTo(m ÷ 4)
        pA = pointer(A) + 4(i - 1) * sizeof(Float64)
        pB = pointer(B) + 4(i - 1) * sB * sizeof(Float64)

        for j = Base.OneTo(n ÷ 4)
            
            v1 = vload(SVec{4,Float64}, pA + 0 * sA * sizeof(Float64))
            v2 = vload(SVec{4,Float64}, pA + 1 * sA * sizeof(Float64))
            v3 = vload(SVec{4,Float64}, pA + 2 * sA * sizeof(Float64))
            v4 = vload(SVec{4,Float64}, pA + 3 * sA * sizeof(Float64))

            w1 = SVec{4,Float64}(v1[1], v2[1], v3[1], v4[1])
            w2 = SVec{4,Float64}(v1[2], v2[2], v3[2], v4[2])
            w3 = SVec{4,Float64}(v1[3], v2[3], v3[3], v4[3])
            w4 = SVec{4,Float64}(v1[4], v2[4], v3[4], v4[4])

            vstore!(pB + 0 * sB * sizeof(Float64), w1)
            vstore!(pB + 1 * sB * sizeof(Float64), w2)
            vstore!(pB + 2 * sB * sizeof(Float64), w3)
            vstore!(pB + 3 * sB * sizeof(Float64), w4)

            pA += 4 * sA * sizeof(Float64)
            pB += 4 * sizeof(Float64)
        end
    end

    return B
end

# todo, clean this madness up
function transpose_avx!(B::AbstractMatrix{Float64}, A::AbstractMatrix{Float64})

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
                v1 = vload(SVec{4,Float64}, pA + 0 * sA * sizeof(Float64))
                v2 = vload(SVec{4,Float64}, pA + 1 * sA * sizeof(Float64))
                v3 = vload(SVec{4,Float64}, pA + 2 * sA * sizeof(Float64))
                v4 = vload(SVec{4,Float64}, pA + 3 * sA * sizeof(Float64))

                w1 = SVec{4,Float64}(v1[1], v2[1], v3[1], v4[1])
                w2 = SVec{4,Float64}(v1[2], v2[2], v3[2], v4[2])
                w3 = SVec{4,Float64}(v1[3], v2[3], v3[3], v4[3])
                w4 = SVec{4,Float64}(v1[4], v2[4], v3[4], v4[4])

                vstore!(pB + 0 * sB * sizeof(Float64), w1)
                vstore!(pB + 1 * sB * sizeof(Float64), w2)
                vstore!(pB + 2 * sB * sizeof(Float64), w3)
                vstore!(pB + 3 * sB * sizeof(Float64), w4)

                pA += 4 * sA * sizeof(Float64)
                pB += 4 * sizeof(Float64)
                j += 4
            end

            while j ≤ n
                B[j, i + 0] = A[i + 0, j]
                B[j, i + 1] = A[i + 1, j]
                B[j, i + 2] = A[i + 2, j]
                B[j, i + 3] = A[i + 3, j]

                j += 1
            end

            i += 4
        end

        while i ≤ m
            pA = pointer(A) + (i - 1) * sizeof(Float64)
            pB = pointer(B) + (i - 1) * sB * sizeof(Float64)

            j = 1
            while j + 4 ≤ n
                B[j + 0, i] = A[i, j + 0]
                B[j + 1, i] = A[i, j + 1]
                B[j + 2, i] = A[i, j + 2]
                B[j + 3, i] = A[i, j + 3]

                j += 4
            end

            while j ≤ n
                B[j, i] = A[i, j]
                j += 1
            end

            i += 1
        end
    end

    return B
end

divisable_by_4(k) = rem(k, 4) == 0

power_of_two_smaller_than_or_equal_to(k::T) where {T} = (1 << (8sizeof(T) - leading_zeros(k) - 1)) % Int

function recursive_transpose!(A::AbstractMatrix, B::AbstractMatrix, b::Val{blocksize} = Val(32)) where blocksize
    m, n = size(A)

    if max(m, n) ≤ blocksize
        if divisable_by_4(m) && divisable_by_4(n)
            transpose_avx_multiples_of_four!(A, B)
        else
            transpose_avx!(A, B)
        end
    else
        k = power_of_two_smaller_than_or_equal_to(max(m, n) ÷ 2)
        if m > n
            recursive_transpose!(view(A, 1:k, :), view(B, :, 1:k), b)
            recursive_transpose!(view(A, k+1:m, :), view(B, :, k+1:m), b)
        else
            recursive_transpose!(view(A, :, 1:k), view(B, 1:k, :), b)
            recursive_transpose!(view(A, :, k+1:n), view(B, k+1:n, :), b)
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