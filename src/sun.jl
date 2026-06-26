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
    structure_constants(G::AbstractVector{<:AbstractMatrix}) -> ConstantSparseTensor

Generic structure constants `fᵃᵇᶜ` for *any* basis of **Hermitian** generators
`G` (physics convention `[Gᵃ,Gᵇ] = i fᵃᵇᶜ Gᶜ`), as a [`ConstantSparseTensor`](@ref).
The basis must be orthogonal under the trace form (the standard su/u/usp bases
are). This is the engine behind every group family — plug in `generators(N)`,
`u_generators(N)`, `sp_generators(n)`, or your own. (For SO(N), whose generators
are real antisymmetric rather than Hermitian, use [`so_structure_constants`](@ref).)
"""
function structure_constants(G::AbstractVector{<:AbstractMatrix})
    M = length(G)
    f = zeros(M, M, M)
    nrm = [real(tr(G[c] * G[c])) for c in 1:M]
    for a in 1:M, b in 1:M
        comm = G[a] * G[b] - G[b] * G[a]                 # = i Σ fᵃᵇᶜ Gᶜ
        for c in 1:M
            f[a, b, c] = real(-im * tr(G[c] * comm)) / nrm[c]
        end
    end
    return ConstantSparseTensor(f)
end

"""
    u_generators(N) -> Vector{SMatrix{N,N,ComplexF64}}

The `N²` Hermitian generators of `u(N) = su(N) ⊕ u(1)`: the SU(N) generators plus
the identity (the `u(1)` direction, which commutes with everything). `exp(i θᵃGᵃ)
∈ U(N)`.
"""
function u_generators(N::Integer)
    G = generators(N)
    push!(G, SMatrix{N,N,ComplexF64}(I))
    return G
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
    groupexp(X::SMatrix) -> SMatrix

Exponential map `X ↦ exp(X)` from the (anti-Hermitian) Lie algebra to the group.
Dispatches to the fast closed form for SU(2) ([`su2_exp`](@ref)) and SU(3)
([`mp_exp`](@ref)); for every other size it falls back to StaticArrays' `exp`,
which is already allocation-free and, for a generic static matrix, faster than a
hand-rolled series. (A general Cayley–Hamilton expansion was benchmarked and is
*not* worth it — `exp` wins for `N ≠ 2, 3`.)
"""
groupexp(X::SMatrix) = exp(X)
# split off the trace (u(1) part) so the closed forms — which assume a traceless,
# unimodular generator — stay correct for u(N)/general matrices, not just su(N).
groupexp(X::SMatrix{2,2,ComplexF64}) = (τ = tr(X) / 2; exp(τ) * su2_exp(X - τ * one(X)))
groupexp(X::SMatrix{3,3,ComplexF64}) = (τ = tr(X) / 3; exp(τ) * mp_exp(X - τ * one(X)))

"""
    su2_exp(X::SMatrix{2,2,ComplexF64}) -> SMatrix{2,2,ComplexF64}

Closed-form SU(2) exponential of an anti-Hermitian traceless `X = iQ`:
`exp(iQ) = cos λ · I + i (sin λ / λ) Q`, where `±λ` are the eigenvalues of the
Hermitian traceless `Q = −iX` (`λ = √(½ Tr Q²)`). The matrix form of Rodrigues —
faster than the generic `exp`.
"""
function su2_exp(X::SMatrix{2,2,ComplexF64})
    Q = -im * X                          # Hermitian, traceless
    λ = sqrt(real(tr(Q * Q)) / 2)        # eigenvalues ±λ
    λ < 1e-12 && return one(X) + im * Q
    return cos(λ) * one(X) + (im * sin(λ) / λ) * Q
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
