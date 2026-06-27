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
