# G₂, the smallest exceptional group, as Aut(𝕆) — the automorphisms of the
# octonions. Its Lie algebra g₂ (14-dimensional) is the derivation algebra: the
# X ∈ so(7) (acting on the 7 imaginary units) satisfying the Leibniz rule for the
# octonion product. We build the octonion structure constants by Cayley–Dickson,
# extract g₂ as the nullspace of the derivation condition inside so(7), and return
# Hermitian generators (i·X). Its structure constants are kept DENSE — at 14 dims
# the @generated ConstantSparseTensor contraction is impractical (compile cost
# ~nnz², nnz≈2300), so g₂ uses the dense `structure_constants_dense` / `casimir` /
# `bracket` / `jacobi_violation` paths.

@inline _qmul(p, q) = SVector(
    p[1] * q[1] - p[2] * q[2] - p[3] * q[3] - p[4] * q[4],
    p[1] * q[2] + p[2] * q[1] + p[3] * q[4] - p[4] * q[3],
    p[1] * q[3] - p[2] * q[4] + p[3] * q[1] + p[4] * q[2],
    p[1] * q[4] + p[2] * q[3] - p[3] * q[2] + p[4] * q[1])
@inline _qconj(q) = SVector(q[1], -q[2], -q[3], -q[4])

# octonion product (Cayley–Dickson doubling of quaternions): (a,b)(c,d) = (ac−d̄b, da+bc̄)
@inline function _omul(x::SVector{8}, y::SVector{8})
    a = SVector(x[1], x[2], x[3], x[4]); b = SVector(x[5], x[6], x[7], x[8])
    c = SVector(y[1], y[2], y[3], y[4]); d = SVector(y[5], y[6], y[7], y[8])
    return vcat(_qmul(a, c) - _qmul(_qconj(d), b), _qmul(d, a) + _qmul(b, _qconj(c)))
end

"""
    octonion_structure_constants() -> Array{Float64,3}

The imaginary-octonion structure constants `c[i,j,k]` (`i,j,k = 1..7`), defined by
`eᵢ eⱼ = −δᵢⱼ + Σₖ c[i,j,k] eₖ`, built by Cayley–Dickson doubling. Totally
antisymmetric (the G₂ associative 3-form).
"""
function octonion_structure_constants()
    c = zeros(7, 7, 7)
    e(i) = setindex(zero(SVector{8,Float64}), 1.0, i)
    for i in 1:7, j in 1:7
        p = _omul(e(i + 1), e(j + 1))
        for k in 1:7
            c[i, j, k] = p[k + 1]
        end
    end
    return c
end

"""
    g2_generators() -> Vector{SMatrix{7,7,ComplexF64}}

The 14 Hermitian generators of `g₂` (the compact exceptional Lie algebra) in its
7-dimensional representation, as the derivation algebra of the octonions inside
so(7). Orthonormal under the trace form.
"""
function g2_generators()
    c = octonion_structure_constants()
    L = [(m = zeros(7, 7); m[a, b] = 1.0; m[b, a] = -1.0; m) for a in 1:7 for b in (a + 1):7]  # so(7), 21
    # derivation condition  Σₖ c[i,j,k] X[m,k] = Σₙ X[n,i] c[n,j,m] + Σₙ X[n,j] c[i,n,m]
    M = zeros(7^3, length(L))
    for (col, Lc) in enumerate(L)
        idx = 0
        for i in 1:7, j in 1:7, m in 1:7
            idx += 1
            v = 0.0
            for k in 1:7; v += c[i, j, k] * Lc[m, k]; end
            for n in 1:7; v -= Lc[n, i] * c[n, j, m] + Lc[n, j] * c[i, n, m]; end
            M[idx, col] = v
        end
    end
    ns = nullspace(M)                                  # 21 × 14, orthonormal columns
    @assert size(ns, 2) == 14 "expected dim g₂ = 14, got $(size(ns, 2))"
    return [SMatrix{7,7,ComplexF64}(im .* sum(ns[a, col] * L[a] for a in eachindex(L)))
            for col in 1:size(ns, 2)]
end

"""
    g2_structure_constants() -> Array{Float64,3}

The `g₂` structure constants `fᵃᵇᶜ` (14-dimensional) as a **dense** array — at this
size the `@generated` [`ConstantSparseTensor`](@ref) contraction is impractical, so
g₂ uses the dense `casimir` / `bracket` / `jacobi_violation` methods.
"""
g2_structure_constants() = structure_constants_dense(g2_generators())
