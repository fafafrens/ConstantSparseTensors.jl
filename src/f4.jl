# F₄ — the last exceptional, non-simply-laced (two root lengths), so the simply-laced
# cocycle doesn't apply. Instead build it as the FIXED SUBALGEBRA of E₆ under its
# diagram automorphism σ (1↔6, 3↔5; 2,4 fixed): F₄ = E₆^σ (dim 52 = 78 − 26). Jacobi
# is inherited (a fixed subalgebra is a subalgebra). We fold the *compact* E₆ so the
# result is compact F₄ with Hermitian generators.

"""
    f4_cartan_matrix() -> Matrix{Int}

The 4×4 Cartan matrix of F₄ (chain 1–2⇒3–4 with the double bond between 2 and 3).
Non-symmetric (`A[2,3]=-2, A[3,2]=-1`), determinant 1.
"""
f4_cartan_matrix() = [2 -1 0 0; -1 2 -2 0; 0 -1 2 -1; 0 0 -1 2]

# the E₆ diagram automorphism Θ as a matrix in the Cartan–Weyl basis
function _e6_diagram_automorphism(C, roots)
    n = 6; D = size(C, 1)
    ri = Dict(α => k for (k, α) in enumerate(roots)); eidx(α) = n + ri[α]
    isroot(α) = haskey(ri, α)
    N(α, β) = isroot(α .+ β) ? C[eidx(α), eidx(β), eidx(α .+ β)] : 0.0
    p = [6, 2, 5, 4, 3, 1]; σ(α) = α[p]                       # σ(α_i)=α_{p[i]}
    c = Dict{Vector{Int},Float64}()                          # Θ(e_α)=c_α e_{σα}
    for i in 1:n; s = zeros(Int, n); s[i] = 1; c[s] = 1.0; c[-s] = 1.0; end
    for γ in sort([α for α in roots if sum(α) > 0], by = sum)
        haskey(c, γ) && continue
        for α in roots, β in roots
            (α .+ β == γ && haskey(c, α) && haskey(c, β)) || continue
            c[γ] = c[α] * c[β] * N(σ(α), σ(β)) / N(α, β); c[-γ] = c[γ]; break
        end
    end
    Θ = zeros(D, D)
    for i in 1:n; Θ[p[i], i] = 1.0; end                      # Θ(h_i)=h_{σi}
    for α in roots; Θ[eidx(σ(α)), eidx(α)] = c[α]; end
    return Θ
end

# compact F₄ structure constants by folding compact E₆ on the +1 eigenspace of Θ
function _f4_structure_constants()
    C, roots = chevalley_structure_constants(e6_cartan_matrix())
    fo, B = _compact_form(C, 6, length(roots) ÷ 2)
    Θ = real(inv(B) * _e6_diagram_automorphism(C, roots) * B)    # Θ in the compact basis
    Q = nullspace(Θ - I)                                         # fixed subalgebra (78×52)
    D = size(fo, 1); m = size(Q, 2)
    tmp = zeros(m, m, D)
    for c in 1:D; tmp[:, :, c] = Q' * fo[:, :, c] * Q; end
    f4 = zeros(m, m, m)
    for k in 1:m, c in 1:D; @views f4[:, :, k] .+= Q[c, k] .* tmp[:, :, c]; end
    return f4
end

"""
    f4_structure_constants() -> Array{Float64,3}

The structure constants of F₄ (dimension 52) as a dense array — the compact, totally
antisymmetric form, obtained by folding E₆ on its diagram automorphism (so Jacobi is
inherited). `jacobi_violation` ≈ 0.
"""
f4_structure_constants() = _f4_structure_constants()

"""
    f4_generators() -> Vector{Matrix{ComplexF64}}

The 52 compact Hermitian generators of F₄ (adjoint representation). `root_system`
recovers the F₄ Cartan matrix — rank 4, 48 roots, the non-simply-laced double bond.
"""
f4_generators() = _hermitian_adjoint(_f4_structure_constants())
