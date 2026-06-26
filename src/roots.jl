# Root-system extraction: recover the rank, the roots, a set of simple roots, and
# the Cartan matrix from a Hermitian generator basis alone — i.e. rediscover the
# algebra's type (Aₙ/Bₙ/Cₙ/Dₙ/…) from the matrices, the way Cartan classified them.
#
# Method: pick a generic Cartan element H; its centralizer (kernel of ad_H) is a
# Cartan subalgebra of rank r. The remaining ad-eigenvectors are root vectors, and
# each root α is read off as the eigenvalues of the CSA acting on it. Simple roots
# and the Cartan matrix follow.

"""
    RootSystem

Result of [`root_system`](@ref): the `rank`, the full list of `roots` (each a
length-`rank` real vector), a choice of `simple` roots, and the integer `cartan`
matrix `Aᵢⱼ = 2⟨αᵢ,αⱼ⟩/⟨αⱼ,αⱼ⟩`.
"""
struct RootSystem
    rank::Int
    roots::Vector{Vector{Float64}}
    simple::Vector{Vector{Float64}}
    cartan::Matrix{Int}
end

# real antisymmetric ad-matrix of  H = Σ θ_a Tᵃ  in the adjoint basis:
# [H, Tᵇ] = i Σ_c R[c,b] Tᶜ,  R[c,b] = Σ_a θ_a f[a,b,c]   (so ad_H = i·R)
_adR(fd, θ) = [sum(θ[a] * fd[a, b, c] for a in axes(fd, 1)) for c in axes(fd, 3), b in axes(fd, 2)]

function _dedupe(vs)
    out = Vector{Float64}[]
    for v in vs
        any(u -> norm(u - v) < 1e-4, out) || push!(out, v)
    end
    return out
end

function _simple_roots(roots)
    r = length(first(roots))
    for _ in 1:50
        ℓ = randn(r)
        minimum(abs(dot(ℓ, α)) for α in roots) < 1e-6 && continue       # α ⟂ ℓ: retry
        pos = [α for α in roots if dot(ℓ, α) > 0]
        issum(α) = any(norm(α - (β + γ)) < 1e-4 for β in pos, γ in pos)
        simple = [α for α in pos if !issum(α)]
        length(simple) == r && return simple
    end
    error("could not isolate simple roots")
end

"""
    root_system(G; tol=1e-7) -> RootSystem

Extract the root system of the (compact, simple) Lie algebra spanned by the
Hermitian generators `G`. Returns the rank, roots, simple roots and Cartan matrix —
recovering the Dynkin type from the matrices alone.

```julia
rs = root_system(generators(3))     # su(3) = A₂
rs.rank          # 2
length(rs.roots) # 6
rs.cartan        # [2 -1; -1 2]
```
"""
function root_system(G::AbstractVector{<:AbstractMatrix}; tol = 1e-7)
    fd = todense(structure_constants(G))
    M = size(fd, 1)
    for _ in 1:50
        R0 = _adR(fd, randn(M))                       # generic Cartan element
        H = nullspace(R0; atol = tol)                 # CSA basis (M × rank), orthonormal
        r = size(H, 2)
        Ri = [_adR(fd, H[:, i]) for i in 1:r]
        F = eigen(R0)
        roots = Vector{Float64}[]
        ok = true
        for (idx, λ) in enumerate(F.values)
            abs(λ) < tol && continue                  # CSA direction, not a root
            E = F.vectors[:, idx]
            α = [real(im * (E' * (Ri[i] * E)) / (E' * E)) for i in 1:r]   # R_i E = −iαᵢ E
            norm(α) < tol && (ok = false; break)      # degenerate eigvec → retry
            push!(roots, α)
        end
        ok || continue
        roots = _dedupe(roots)
        length(roots) == M - r || continue            # all root spaces 1-dimensional
        simple = _simple_roots(roots)
        A = [round(Int, 2 * dot(simple[i], simple[j]) / dot(simple[j], simple[j]))
             for i in eachindex(simple), j in eachindex(simple)]
        return RootSystem(r, roots, simple, A)
    end
    error("root-system extraction failed (degenerate Cartan element)")
end
