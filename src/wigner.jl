# Wigner D-matrices (the action of a group element in a representation) and the
# Haar average — which by the averaging theorem is the projector onto invariants.

"""
    wigner(G, θ) -> Matrix

The Wigner D-matrix: the representation of the group element `exp(i θᵃ Tᵃ)` in the
representation with generators `G` (`= groupexp(algebra(θ, G))`). Unitary.
"""
wigner(G::AbstractVector{<:AbstractMatrix}, θ::AbstractVector) =
    groupexp(algebra(SVector{length(θ)}(θ), G))

"""
    invariant_projector(G) -> Matrix

The orthogonal projector onto the invariant (trivial) subspace of the representation
`G`: the common kernel `∩ₐ ker Tᵃ` of the generators. By the averaging theorem this
equals the Haar integral `∫_G ρ(g) dg`; its rank is the multiplicity of the trivial
representation (so it is `0` for a nontrivial irrep).
"""
function invariant_projector(G::AbstractVector{<:AbstractMatrix})
    Q = nullspace(reduce(vcat, [Matrix(g) for g in G]))
    return Q * Q'
end

"""
    haar_average(G; samples=3000, steps=12) -> Matrix

Monte-Carlo estimate of the Haar integral `∫_G ρ(g) dg`, averaging the Wigner matrix
over group elements drawn by a random walk (products of `steps` random
exponentials). Converges to [`invariant_projector`](@ref) — a numerical check of the
theorem that the group average is the projector onto invariants.
"""
function haar_average(G::AbstractVector{<:AbstractMatrix}; samples = 3000, steps = 12)
    M = length(G); D = size(G[1], 1)
    acc = zeros(ComplexF64, D, D)
    for _ in 1:samples
        U = Matrix{ComplexF64}(I, D, D)
        for _ in 1:steps
            U = groupexp(algebra(SVector{M}(randn(M)), G)) * U
        end
        acc .+= U
    end
    return acc ./ samples
end

"""
    haar_sample(G; steps=12) -> Matrix

A Haar-random group element in the representation `G`, drawn by a random walk
(product of `steps` random exponentials). Exposes the Haar measure: averaging any
function of `haar_sample(G)` estimates its Haar integral.
"""
function haar_sample(G::AbstractVector{<:AbstractMatrix}; steps = 12)
    M = length(G); D = size(G[1], 1)
    U = Matrix{ComplexF64}(I, D, D)
    for _ in 1:steps
        U = groupexp(algebra(SVector{M}(randn(M)), G)) * U
    end
    return U
end

"""
    trivial_rep(G) -> Vector

The trivial (1-dimensional) representation of the same algebra as `G`: every
generator is zero. `character_projector(V, trivial_rep(V))` is
[`invariant_projector`](@ref).
"""
trivial_rep(G::AbstractVector{<:AbstractMatrix}) = [zeros(ComplexF64, 1, 1) for _ in G]

"""
    character_projector(V, R; samples=4000, steps=12) -> Matrix

The projector onto the `R`-isotypic component of the representation `V`, via the
character projection `P_R = d_R ∫_G conj(χ_R(g)) ρ_V(g) dg`, with `d_R = dim R` and
`χ_R(g) = Tr ρ_R(g)`. Estimated by Monte-Carlo over a shared Haar random walk
evaluated in both reps. Generalizes [`invariant_projector`](@ref) (the case where
`R` is trivial, `χ_R ≡ 1`).
"""
function character_projector(V::AbstractVector{<:AbstractMatrix},
                             R::AbstractVector{<:AbstractMatrix}; samples = 4000, steps = 12)
    M = length(V)
    @assert length(R) == M "V and R must be representations of the same algebra"
    DV = size(V[1], 1); dR = size(R[1], 1)
    acc = zeros(ComplexF64, DV, DV)
    for _ in 1:samples
        UV = Matrix{ComplexF64}(I, DV, DV)
        UR = Matrix{ComplexF64}(I, dR, dR)
        for _ in 1:steps
            θ = SVector{M}(randn(M))
            UV = groupexp(algebra(θ, V)) * UV
            UR = groupexp(algebra(θ, R)) * UR        # same g, evaluated in R
        end
        acc .+= conj(tr(UR)) * UV                     # conj(χ_R(g)) · ρ_V(g)
    end
    return (dR / samples) * acc
end
