# SU(N) Lie algebra as an application of ConstantSparseTensor: the structure
# constants fᵃᵇᶜ (antisymmetric) and dᵃᵇᶜ (symmetric) for arbitrary N, plus the
# generators, the exponential map (algebra → group), and the bracket / Casimir /
# adjoint representation that all fall out of one `contract` primitive.

"""
    gellmann(N) -> Vector{Matrix{ComplexF64}}

The `N²−1` generalized Gell-Mann matrices, normalized `Tr(λᵃλᵇ) = 2δᵃᵇ`.
SU(2): the Pauli matrices. SU(3): the eight Gell-Mann matrices.
"""
function gellmann(N::Integer)
    Λ = Matrix{ComplexF64}[]
    for i in 1:N, j in (i + 1):N                       # off-diagonal: symmetric + antisymmetric
        s = zeros(ComplexF64, N, N); s[i, j] = 1;  s[j, i] = 1;   push!(Λ, s)
        a = zeros(ComplexF64, N, N); a[i, j] = -im; a[j, i] = im; push!(Λ, a)
    end
    for l in 1:(N - 1)                                  # diagonal
        d = zeros(ComplexF64, N, N)
        for k in 1:l; d[k, k] = 1; end
        d[l + 1, l + 1] = -l
        push!(Λ, d .* sqrt(2 / (l * (l + 1))))
    end
    return Λ
end

"""
    generators(N) -> Vector{SMatrix{N,N,ComplexF64}}

The fundamental-representation generators `Tᵃ = λᵃ/2` as `SMatrix`es
(Hermitian, traceless, `Tr(TᵃTᵇ) = δᵃᵇ/2`).
"""
generators(N::Integer) = [SMatrix{N,N,ComplexF64}(λ ./ 2) for λ in gellmann(N)]

# dense f and d (dim M = N²−1) from the generators via the trace formulas
#   fᵃᵇᶜ = (1/4i) Tr([λᵃ,λᵇ] λᶜ)   (totally antisymmetric)
#   dᵃᵇᶜ = (1/4)  Tr({λᵃ,λᵇ} λᶜ)   (totally symmetric)
function _structure_constants_dense(N::Integer)
    Λ = gellmann(N); M = N^2 - 1
    f = zeros(M, M, M); d = zeros(M, M, M)
    for a in 1:M, b in 1:M
        comm = Λ[a] * Λ[b] - Λ[b] * Λ[a]
        anti = Λ[a] * Λ[b] + Λ[b] * Λ[a]
        for c in 1:M
            f[a, b, c] = real(-im * tr(comm * Λ[c]) / 4)
            d[a, b, c] = real(tr(anti * Λ[c]) / 4)
        end
    end
    return f, d
end

"""
    structure_constants(N) -> (f, d)

The SU(N) structure constants as [`ConstantSparseTensor`](@ref)s: the totally
antisymmetric `fᵃᵇᶜ` and totally symmetric `dᵃᵇᶜ` (both rank-3, dimension
`M = N²−1`). SU(2): `f` is the Levi-Civita ε and `d ≡ 0`.
"""
function structure_constants(N::Integer)
    f, d = _structure_constants_dense(N)
    return ConstantSparseTensor(f), ConstantSparseTensor(d)
end

"""
    bracket(f, x, y) -> SVector

The Lie bracket on coefficient vectors: `[x,y]ᶜ = fᵃᵇᶜ xᵃ yᵇ` (`= tdot(f, x, y)`).
For `X = xᵃTᵃ`, `Y = yᵃTᵃ`, this returns the components `c` of `[X,Y] = i cᵃTᵃ`.
"""
bracket(f::ConstantSparseTensor, x::SVector, y::SVector) = tdot(f, x, y)

"""
    casimir(f) -> ConstantSparseTensor

The adjoint Casimir `Cᵉᵍ = Σ_{ab} fᵃᵇᵉ fᵃᵇᵍ = N δᵉᵍ`, via [`contract`](@ref).
"""
casimir(f::ConstantSparseTensor) = contract(f, f, Val(((1, 1), (2, 2))))

"""
    adjoint_generators(N) -> Vector{SMatrix{M,M,ComplexF64}}

The adjoint-representation generators `(Tᵃ_adj)_bc = −i fᵃᵇᶜ` (`M = N²−1`). They
satisfy the same algebra `[Tᵃ,Tᵇ] = i fᵃᵇᶜ Tᶜ` and `Tr(Tᵃ_adj Tᵇ_adj) = N δᵃᵇ`.
"""
function adjoint_generators(N::Integer)
    f, _ = _structure_constants_dense(N)
    M = N^2 - 1
    return [SMatrix{M,M,ComplexF64}(-im * f[a, b, c] for b in 1:M, c in 1:M) for a in 1:M]
end

# ---- exponential map: algebra → group --------------------------------------

"""
    algebra(θ::SVector, T) -> SMatrix

Build the Lie-algebra element `X(θ) = i θᵃ Tᵃ` (anti-Hermitian, traceless) from
real parameters `θ` and a generator list `T` (e.g. from [`generators`](@ref)).
`exp(X)` is then a group element of SU(N). This is the lattice gauge-link /
HMC update `U = exp(i ε πᵃ Tᵃ)`.
"""
algebra(θ::SVector, T) = sum(im * θ[a] * T[a] for a in eachindex(θ))

"""
    su2_exp(θ::SVector{3}) -> SMatrix{2,2,ComplexF64}

Closed-form (Rodrigues) SU(2) exponential
`exp(i θ·σ/2) = cos(|θ|/2) I + i sin(|θ|/2) (θ̂·σ)`.
"""
function su2_exp(θ::SVector{3})
    σ1 = SA[0.0+0im 1; 1 0]; σ2 = SA[0.0 -im; im 0]; σ3 = SA[1.0+0im 0; 0 -1]
    Id = SMatrix{2,2,ComplexF64}(I)
    n  = sqrt(θ[1]^2 + θ[2]^2 + θ[3]^2)
    n < 1e-14 && return Id
    h = n / 2
    return cos(h) * Id + (im * sin(h) / n) * (θ[1] * σ1 + θ[2] * σ2 + θ[3] * σ3)
end

# one companion-matrix step  a ← X·a  reduced via  X^N = Σ b[k] X^{k-1}.
# A standalone function so `a` is an argument (never a captured, reassigned closure
# variable — that boxing is what kills performance and allocates).
@inline _companion(a::SVector{N,T}, b::SVector{N,T}) where {N,T} =
    SVector{N,T}(ntuple(k -> (k == 1 ? zero(T) : a[k - 1]) + a[N] * b[k], Val(N)))

"""
    expch(X::SMatrix{N,N}; nterms=20) -> SMatrix{N,N}

Matrix exponential at near-closed-form speed via Cayley–Hamilton:
`exp(X) = Σ_{k=0}^{N-1} c_k X^k`. The `c_k` come from the characteristic
polynomial (Faddeev–LeVerrier) plus a Taylor recurrence — no eigendecomposition,
no scaling/squaring. Same code for every `N`. For large `‖X‖`, raise `nterms`.
"""
function expch(X::SMatrix{N,N,T}; nterms::Int = 20) where {N,T}
    Id = one(X)

    # 1. characteristic polynomial via Faddeev–LeVerrier  →  X^N = Σ_k b[k] X^{k-1}
    Mk  = Id
    csv = zero(MVector{N,T})                       # csv[i] = coeff of X^{i-1} in char poly
    for k in 1:N
        Mk  = X * Mk
        ck  = -tr(Mk) / k
        csv[N + 1 - k] = ck
        Mk  = Mk + ck * Id
    end
    b = -SVector{N,T}(csv)                              # reduction: X^N = Σ b[k] X^{k-1}

    # 2. Taylor series kept in the basis {I,X,…,X^{N-1}} via the companion step
    a = SVector{N,T}(ntuple(k -> k == 1 ? one(T) : zero(T), Val(N)))   # X^0 = I
    c = a                                                             # /0!
    invf = 1.0
    for m in 1:nterms
        invf /= m
        a = _companion(a, b)                            # a ← X·a (reduced) — no boxing
        c = c + a * invf
    end

    # 3. reassemble  Σ c[k] X^{k-1}  by Horner
    R = c[N] * Id
    for k in (N - 1):-1:1
        R = R * X + c[k] * Id
    end
    return R
end

"""
    mp_exp(X::SMatrix{3,3,ComplexF64}) -> SMatrix{3,3,ComplexF64}

Morningstar–Peardon closed form for SU(3): `exp(X) = exp(iQ) = f₀I + f₁Q + f₂Q²`
with the `fⱼ` in closed form from the two invariants of the Hermitian traceless
`Q = −iX` (`c₁ = ½TrQ²`, `c₀ = detQ`). No eigensolve, no recurrence — a handful
of transcendentals + two matmuls. Fastest exact SU(3) exponential.
"""
function mp_exp(X::SMatrix{3,3,ComplexF64})
    Q  = -im * X                              # Hermitian, traceless
    c1 = real(tr(Q * Q)) / 2                  # ½ Tr Q²   (invariant)
    c1 < 1e-12 && return one(X) + im * Q      # Q ≈ 0
    c0   = real(det(Q))                       # det Q = ⅓ Tr Q³   (invariant)
    s    = c0 < 0 ? -1 : 1                     # use the c0≥0 branch, then symmetry
    c0m  = 2 * (c1 / 3)^(3 / 2)                # c0_max
    θ    = acos(clamp(abs(c0) / c0m, -1.0, 1.0)) / 3
    u    = sqrt(c1 / 3) * cos(θ)               # u, w parameterize the 3 eigenvalues
    w    = sqrt(c1) * sin(θ)
    ξ0   = abs(w) < 1e-4 ? 1 - w^2 / 6 * (1 - w^2 / 20 * (1 - w^2 / 42)) : sin(w) / w
    cw   = cos(w)
    e2iu = exp(2im * u); emiu = exp(-im * u)
    den  = 9u^2 - w^2
    f0 = ((u^2 - w^2) * e2iu + emiu * (8u^2 * cw + 2im * u * (3u^2 + w^2) * ξ0)) / den
    f1 = (2u * e2iu - emiu * (2u * cw - im * (3u^2 - w^2) * ξ0)) / den
    f2 = (e2iu - emiu * (cw + 3im * u * ξ0)) / den
    if s < 0                                   # fⱼ(−c0) = (−1)ʲ conj(fⱼ(c0))
        f0, f1, f2 = conj(f0), -conj(f1), conj(f2)
    end
    return f0 * one(X) + f1 * Q + f2 * (Q * Q)
end
