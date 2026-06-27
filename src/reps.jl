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

"""
    Irrep

One isotypic component from [`decompose`](@ref): an irreducible of dimension `dim`
appearing with `multiplicity`, at quadratic-Casimir value `casimir`, spanning the
columns of `basis` (an orthonormal `D × (dim·multiplicity)` block of the rep space).
"""
struct Irrep
    dim::Int
    multiplicity::Int
    casimir::Float64
    basis::Matrix{ComplexF64}
end

# dimension of the commutant of the generators restricted to the subspace P:
# {M : [M, P†TᵃP] = 0 ∀a}. By Schur this is m² for m copies of one irreducible.
function _commutant_dim(G, P; tol)
    dW = size(P, 2)
    Id = Matrix{ComplexF64}(I, dW, dW)
    K = reduce(vcat, [(t = P' * Matrix(g) * P; kron(transpose(t), Id) - kron(Id, t)) for g in G])
    return size(nullspace(K; atol = tol), 2)
end

"""
    decompose(G; tol=1e-6) -> Vector{Irrep}

Decompose the (possibly reducible) representation `G` into irreducibles. The
quadratic Casimir `C₂ = Σ TᵃTᵃ` is central, so its eigenspaces are invariant; each
distinct eigenvalue gives an isotypic component, and the multiplicity follows from
the dimension of the commutant there (Schur: `dim = m²`). Multiplicity-free tensor
products (e.g. `3⊗3̄ = 8⊕1`) are fully resolved; the method separates irreps by
their Casimir value (it cannot split two *distinct* irreps that happen to share one).

```julia
decompose(tensor_rep(generators(3), conjugate_rep(generators(3))))  # 8 ⊕ 1
decompose(direct_sum_rep(generators(3), generators(3)))             # 3 with multiplicity 2
```
"""
function decompose(G::AbstractVector{<:AbstractMatrix}; tol = 1e-6)
    F = eigen(Hermitian(Matrix(quadratic_casimir(G))))
    D = length(F.values)
    out = Irrep[]
    i = 1
    while i <= D
        j = i
        while j < D && abs(F.values[j + 1] - F.values[i]) < tol
            j += 1
        end
        P = F.vectors[:, i:j]
        mult = round(Int, sqrt(_commutant_dim(G, P; tol)))
        push!(out, Irrep((j - i + 1) ÷ mult, mult, F.values[i], P))
        i = j + 1
    end
    return out
end
