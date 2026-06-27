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

# basis of the commutant  {M : [M, op] = 0 for every op in `ops`}, as D×D matrices.
function _commutant(ops; tol)
    D = size(first(ops), 1)
    Id = Matrix{ComplexF64}(I, D, D)
    K = reduce(vcat, [(o = Matrix(op); kron(transpose(o), Id) - kron(Id, o)) for op in ops])
    ns = nullspace(K; atol = tol)
    return [reshape(ns[:, k], D, D) for k in axes(ns, 2)]
end

"""
    decompose(G; tol=1e-6) -> Vector{Irrep}

Decompose the (possibly reducible) representation `G` into irreducibles. The
commutant `𝒞 = {M : [M,Tᵃ]=0}` is `⊕ᵢ M_{mᵢ}(ℂ)` (Schur), so a generic element of
its **center** separates the isotypic components — distinguishing *all* inequivalent
irreps, even ones with equal Casimir (e.g. a rep and its conjugate). The
multiplicity `mᵢ` of each component then follows from the commutant dimension there
(`= mᵢ²`). Fully general — no Casimir tower needed.

```julia
decompose(tensor_rep(generators(3), conjugate_rep(generators(3))))  # 1 ⊕ 8
decompose(direct_sum_rep(generators(3), generators(3)))             # 3, multiplicity 2
decompose(direct_sum_rep(generators(3), conjugate_rep(generators(3))))  # 3 and 3̄ (two dim-3 irreps)
```
"""
function decompose(G::AbstractVector{<:AbstractMatrix}; tol = 1e-6)
    Cb = _commutant(G; tol)                                   # commutant 𝒞
    Zb = _commutant(vcat(collect(G), Cb); tol)               # center of 𝒞 (one dim / irrep type)
    z = sum(randn(ComplexF64) * Z for Z in Zb)
    F = eigen(Hermitian(Matrix(z + z')))                     # generic Hermitian central element
    C2 = Matrix(quadratic_casimir(G))
    D = length(F.values)
    out = Irrep[]
    i = 1
    while i <= D
        j = i
        while j < D && abs(F.values[j + 1] - F.values[i]) < tol
            j += 1
        end
        P = F.vectors[:, i:j]                                 # one isotypic component
        m = round(Int, sqrt(length(_commutant([P' * Matrix(g) * P for g in G]; tol))))
        push!(out, Irrep((j - i + 1) ÷ m, m, real(tr(P' * C2 * P)) / (j - i + 1), P))
        i = j + 1
    end
    return out
end
