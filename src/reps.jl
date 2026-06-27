# Building representations from generators. The fundamental is `generators(N)`; from
# it `conjugate_rep`/`tensor_rep`/`direct_sum_rep` build the others (3̄, 3⊗3̄, …),
# which then plug straight into weights / quadratic_casimir / structure_constants.

"""
    conjugate_rep(G) -> Vector{Matrix}

The complex-conjugate representation: generators `−conj(Gᵃ)`. For SU(3) this turns
the fundamental **3** (`generators(3)`) into the anti-fundamental **3̄**. Same
structure constants and quadratic Casimir as `G`; the weights are negated.
"""
conjugate_rep(G::AbstractVector{<:AbstractMatrix}) = [-conj(Matrix(g)) for g in G]

"""
    tensor_rep(A, B) -> Vector{Matrix}

The tensor-product representation `A ⊗ B`: generators `Aᵃ⊗I + I⊗Bᵃ`, dimension
`dim(A)·dim(B)`. Generally reducible — e.g. for SU(3), `tensor_rep(3, 3̄)` is the
9-dimensional `3 ⊗ 3̄ = 8 ⊕ 1` (its quadratic Casimir has eigenvalue 3 on the 8 and
0 on the singlet).
"""
function tensor_rep(A::AbstractVector{<:AbstractMatrix}, B::AbstractVector{<:AbstractMatrix})
    @assert length(A) == length(B) "representations of the same algebra"
    da = size(A[1], 1); db = size(B[1], 1)
    Ia = Matrix{ComplexF64}(I, da, da); Ib = Matrix{ComplexF64}(I, db, db)
    return [kron(Matrix{ComplexF64}(a), Ib) + kron(Ia, Matrix{ComplexF64}(b)) for (a, b) in zip(A, B)]
end

"""
    direct_sum_rep(A, B) -> Vector{Matrix}

The direct-sum representation `A ⊕ B`: block-diagonal generators `diag(Aᵃ, Bᵃ)`,
dimension `dim(A) + dim(B)`.
"""
function direct_sum_rep(A::AbstractVector{<:AbstractMatrix}, B::AbstractVector{<:AbstractMatrix})
    @assert length(A) == length(B) "representations of the same algebra"
    return [cat(Matrix{ComplexF64}(a), Matrix{ComplexF64}(b); dims = (1, 2)) for (a, b) in zip(A, B)]
end
