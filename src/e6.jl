# Simply-laced Lie algebras via the Chevalley construction — used here for E₆ (the
# first exceptional beyond g₂). From a simply-laced Cartan matrix: generate the root
# system, fix the structure-constant signs with a 2-cocycle on the root lattice, and
# assemble the structure constants in the Cartan–Weyl basis [h₁..hₙ, e_α]. The result
# satisfies the Jacobi identity (the oracle for the signs). Kept DENSE — at dim 78 the
# @generated ConstantSparseTensor contraction is impractical (as for g₂).
#
# The same `chevalley_structure_constants` builds any simply-laced algebra (Aₙ, Dₙ,
# E₆/E₇/E₈) — only the Cartan matrix changes.

"""
    e6_cartan_matrix() -> Matrix{Int}

The 6×6 Cartan matrix of E₆ (Bourbaki labelling: chain 1–3–4–5–6 with node 2
attached to node 4).
"""
function e6_cartan_matrix()
    A = fill(0, 6, 6)
    for i in 1:6; A[i, i] = 2; end
    for (i, j) in ((1, 3), (3, 4), (4, 5), (5, 6), (2, 4)); A[i, j] = A[j, i] = -1; end
    return A
end

# positive roots of a simply-laced Cartan matrix, as integer coefficient vectors
function _simply_laced_positive_roots(A)
    n = size(A, 1)
    unit(i) = (v = zeros(Int, n); v[i] = 1; v)
    simple = [unit(i) for i in 1:n]
    roots = Set(simple); pos = copy(simple); frontier = copy(simple)
    while !isempty(frontier)
        newf = Vector{Int}[]
        for β in frontier, i in 1:n
            p = 0
            while (β .- (p + 1) .* simple[i]) in roots; p += 1; end
            if p - dot(A[i, :], β) ≥ 1                       # α_i-string: β+α_i is a root
                γ = β .+ simple[i]
                if !(γ in roots); push!(roots, γ); push!(newf, γ); push!(pos, γ); end
            end
        end
        frontier = newf
    end
    return pos
end

"""
    chevalley_structure_constants(A) -> (C, roots)

Structure constants of the simply-laced Lie algebra with (symmetric, simply-laced)
Cartan matrix `A`, in the Cartan–Weyl basis `[h₁..hₙ, e_α]`. `C[a,b,c]` is the
coefficient of basis element `c` in `[Xₐ, X_b]` (real, antisymmetric in `a,b`,
satisfies Jacobi). `roots` is the list of root coefficient vectors; the dimension is
`n + length(roots)`. Structure-constant signs come from a 2-cocycle on the root
lattice (orient each Dynkin edge low→high).
"""
function chevalley_structure_constants(A::AbstractMatrix{<:Integer})
    n = size(A, 1)
    roots = let pos = _simply_laced_positive_roots(A); vcat(pos, [-α for α in pos]); end
    D = n + length(roots)
    ri = Dict(α => k for (k, α) in enumerate(roots))
    isroot(α) = haskey(ri, α)
    eidx(α) = n + ri[α]
    z = zeros(Int, n)
    # sign cocycle  ε_ij: diagonal −1; oriented edge (i<j adjacent) ε_ij=−1, ε_ji=+1; else +1
    ε = fill(1, n, n)
    for i in 1:n; ε[i, i] = -1; end
    for i in 1:n, j in 1:n
        if i < j && A[i, j] == -1
            ε[i, j] = -1; ε[j, i] = 1
        end
    end
    cocycle(α, β) = iseven(sum(ε[i, j] == -1 ? α[i] * β[j] : 0 for i in 1:n, j in 1:n)) ? 1.0 : -1.0
    C = zeros(D, D, D)
    for α in roots, i in 1:n                                # [h_i, e_α] = ⟨α,α_i^∨⟩ e_α
        c = dot(A[i, :], α)
        C[i, eidx(α), eidx(α)] += c
        C[eidx(α), i, eidx(α)] -= c
    end
    for α in roots, β in roots
        if α .+ β == z                                      # [e_α, e_{-α}] = −h_α
            for k in 1:n; C[eidx(α), eidx(β), k] -= α[k]; end
        elseif isroot(α .+ β)                               # [e_α, e_β] = N_{αβ} e_{α+β}
            C[eidx(α), eidx(β), eidx(α .+ β)] += cocycle(α, β)
        end
    end
    return C, roots
end

"""
    e6_structure_constants() -> Array{Float64,3}

The structure constants of E₆ (dimension 78) as a dense array, via
[`chevalley_structure_constants`](@ref) on [`e6_cartan_matrix`](@ref). Satisfies the
Jacobi identity — `jacobi_violation(e6_structure_constants()) ≈ 0`.
"""
e6_structure_constants() = first(chevalley_structure_constants(e6_cartan_matrix()))

"""
    e6_roots() -> Vector{Vector{Int}}

The 72 roots of E₆ as integer coefficient vectors in the simple-root basis (all of
length² = 2).
"""
e6_roots() = last(chevalley_structure_constants(e6_cartan_matrix()))

# Compact Hermitian adjoint generators from split Cartan–Weyl structure constants:
# pass to the compact real form {i h_j, e_α+e_{-α}, i(e_α-e_{-α})} (the f_α=-e_{-α}
# relabeling matches the Chevalley sign convention), orthonormalize under −Killing,
# then Gᵃ_{bc} = −i f_ortho[a,b,c] is Hermitian. roots are ordered [pos; -pos].
function _compact_hermitian_generators(C, n, npos)
    D = size(C, 1)
    P = zeros(ComplexF64, D, D)
    for j in 1:n; P[j, j] = im; end                          # i h_j
    col = n
    for k in 1:npos
        eα = n + k; emα = n + npos + k
        col += 1; P[eα, col] = 1;  P[emα, col] = 1           # e_α + e_{-α}
        col += 1; P[eα, col] = im; P[emα, col] = -im         # i(e_α - e_{-α})
    end
    Pinv = inv(P)
    fc = zeros(ComplexF64, D, D, D)                          # compact structure constants
    for r in 1:D
        Mr = transpose(P) * C[:, :, r] * P
        for c in 1:D; @views fc[:, :, c] .+= Pinv[c, r] .* Mr; end
    end
    fr = real(fc)
    ad = [permutedims(fr[a, :, :], (2, 1)) for a in 1:D]
    K = [tr(ad[a] * ad[b]) for a in 1:D, b in 1:D]           # Killing form (neg. definite)
    Eg = eigen(Symmetric(-K))
    S = Eg.vectors * Diagonal(sqrt.(Eg.values)) * Eg.vectors'
    Sinv = inv(S)
    tmp = similar(fr)
    for r in 1:D; tmp[:, :, r] = Sinv' * fr[:, :, r] * Sinv; end
    fo = zeros(D, D, D)                                       # orthonormal ⇒ totally antisym
    for r in 1:D, c in 1:D; @views fo[:, :, c] .+= S[r, c] .* tmp[:, :, r]; end
    return [ComplexF64[-im * fo[a, b, c] for b in 1:D, c in 1:D] for a in 1:D]
end

"""
    e6_generators() -> Vector{Matrix{ComplexF64}}

The 78 Hermitian generators of E₆ in its adjoint representation (the compact real
form, orthonormalized under the Killing form). `root_system(e6_generators())`
recovers the E₆ Cartan matrix; `casimir`, `decompose`, `weights` all apply.
"""
function e6_generators()
    C, roots = chevalley_structure_constants(e6_cartan_matrix())
    return _compact_hermitian_generators(C, 6, length(roots) ÷ 2)
end
