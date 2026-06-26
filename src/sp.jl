# Sp: the compact symplectic group USp(2n) (Cartan type Cₙ). Its algebra usp(2n)
# is the Hermitian 2n×2n matrices T with TᵀΩ + ΩT = 0, Ω = [0 I; −I 0]. In n×n
# blocks T = [A B; B† −Aᵀ] with A Hermitian and B complex-symmetric — dimension
# n(2n+1). exp(i θᵃTᵃ) ∈ USp(2n) (unitary AND symplectic). usp(2) = su(2).

"""
    symplectic_form(n) -> SMatrix{2n,2n}

The standard symplectic form `Ω = [0 Iₙ; −Iₙ 0]`.
"""
function symplectic_form(n::Integer)
    N = 2n
    Ω = zeros(N, N)
    for i in 1:n
        Ω[i, n + i] = 1.0; Ω[n + i, i] = -1.0
    end
    return SMatrix{N,N,Float64}(Ω)
end

"""
    sp_generators(n) -> Vector{SMatrix{2n,2n,ComplexF64}}

The `n(2n+1)` Hermitian generators of `usp(2n)` (compact symplectic, type Cₙ),
built block-wise `T = [A B; B† −Aᵀ]` from a Hermitian `A`-block (a `u(n)` basis)
and a complex-symmetric `B`-block. Orthogonal under the trace form. `usp(2)` is the
SU(2) generators.
"""
function sp_generators(n::Integer)
    N = 2n
    hbasis = gellmann(n)                                  # su(n) Hermitian basis
    push!(hbasis, Matrix{ComplexF64}(I, n, n))           # complete to u(n)
    G = SMatrix{N,N,ComplexF64}[]
    for h in hbasis                                      # A-block: T = [h 0; 0 −hᵀ]
        T = zeros(ComplexF64, N, N)
        T[1:n, 1:n] = h
        T[n+1:N, n+1:N] = -transpose(h)
        push!(G, SMatrix{N,N,ComplexF64}(T))
    end
    for i in 1:n, j in i:n                               # B-block: complex-symmetric
        S = zeros(n, n); S[i, j] = 1.0; S[j, i] = 1.0
        for B in (ComplexF64.(S), im .* S)               # real- and imaginary-symmetric
            T = zeros(ComplexF64, N, N)
            T[1:n, n+1:N] = B
            T[n+1:N, 1:n] = B'
            push!(G, SMatrix{N,N,ComplexF64}(T))
        end
    end
    # normalize to a uniform trace-norm Tr(g²) = 2 (orthonormal basis) so the
    # structure constants are totally antisymmetric and the Casimir is ∝ I.
    return [g * sqrt(2 / real(tr(g * g))) for g in G]
end

"""
    sp_structure_constants(n) -> ConstantSparseTensor

The `usp(2n)` structure constants `fᵃᵇᶜ` (dimension `n(2n+1)`) as a
[`ConstantSparseTensor`](@ref), via the generic [`structure_constants`](@ref) on
[`sp_generators`](@ref). Pairs with [`bracket`](@ref), [`casimir`](@ref).
"""
sp_structure_constants(n::Integer) = structure_constants(sp_generators(n))
