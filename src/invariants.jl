# Standard invariants that fall out of the generators and structure constants.

"""
    quadratic_casimir(G) -> Matrix

The quadratic Casimir of the representation spanned by the generators `G`,
`C₂ = Σₐ Gᵃ Gᵃ`. By Schur's lemma this is `c·I` on an irreducible representation
(`c = (N²−1)/2N` for the fundamental of su(N)).
"""
quadratic_casimir(G::AbstractVector{<:AbstractMatrix}) = sum(g * g for g in G)

"""
    dynkin_index(G) -> Real

The index `T(R)` of the representation, defined by `Tr(Gᵃ Gᵇ) = T(R) δᵃᵇ` (assumes
an orthogonal generator basis). `T = 1/2` for the fundamental of su(N).
"""
dynkin_index(G::AbstractVector{<:AbstractMatrix}) = real(tr(G[1] * G[1]))

"""
    killing_form(f) -> Matrix

The Killing form `κᵃᵇ = Σ_{cd} fᵃᶜᵈ fᵇᶜᵈ` as a dense matrix (`= todense(casimir(f))`).
Proportional to the identity for a simple algebra in an orthonormal basis.
"""
killing_form(f::ConstantSparseTensor) = todense(casimir(f))

"""
    adjoint_action(f, x) -> Matrix

The matrix of `ad_X = [X, ·]` in the algebra basis, for `X = Σ xₐ Tᵃ`:
`adjoint_action(f, x) * y == bracket(f, x, y)`. The adjoint group element is then
`exp(adjoint_action(f, x))`.
"""
function adjoint_action(f::ConstantSparseTensor, x::AbstractVector)
    fd = todense(f); M = size(fd, 1)
    return [sum(fd[i, j, k] * x[j] for j in 1:M) for i in 1:M, k in 1:M]
end
