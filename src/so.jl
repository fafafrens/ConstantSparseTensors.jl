# SO(N) / O(N): the Lie algebra so(N) is the real ANTISYMMETRIC N×N matrices, so
# the same ConstantSparseTensor structure-constant machinery applies. The standard
# basis L^{(ij)} (i<j), (L^{(ij)})_{kl} = δ_{ik}δ_{jl} − δ_{il}δ_{jk}, gives
# M = N(N−1)/2 generators with integer structure constants. so(3) ≅ su(2): its f is
# the Levi-Civita ε. exp of a real antisymmetric matrix is an SO(N) rotation; O(N)
# adds the det = −1 reflections, which exp cannot reach.

"""
    so_generators(N) -> Vector{SMatrix{N,N,Float64}}

The `M = N(N−1)/2` antisymmetric basis generators of `so(N)`,
`(L^{(ij)})_{kl} = δ_{ik}δ_{jl} − δ_{il}δ_{jk}` for `i < j`, orthonormal under
`⟨A,B⟩ = −½ Tr(AB)`.
"""
function so_generators(N::Integer)
    G = SMatrix{N,N,Float64}[]
    for i in 1:N, j in (i + 1):N
        L = zeros(N, N); L[i, j] = 1.0; L[j, i] = -1.0
        push!(G, SMatrix{N,N,Float64}(L))
    end
    return G
end

function _so_structure_constants_dense(N::Integer)
    G = so_generators(N); M = length(G)
    f = zeros(M, M, M)
    for a in 1:M, b in 1:M
        comm = G[a] * G[b] - G[b] * G[a]
        for c in 1:M
            f[a, b, c] = -tr(comm * G[c]) / 2          # ⟨[Lᵃ,Lᵇ], Lᶜ⟩
        end
    end
    return f
end

"""
    so_structure_constants(N) -> ConstantSparseTensor

The `so(N)` structure constants `fᵃᵇᶜ` (totally antisymmetric, dimension
`M = N(N−1)/2`) as a [`ConstantSparseTensor`](@ref). `so(2)` is abelian (`f ≡ 0`);
`so(3)` gives the Levi-Civita ε. Pairs with [`bracket`](@ref), [`casimir`](@ref).
"""
so_structure_constants(N::Integer) = ConstantSparseTensor(_so_structure_constants_dense(N))

"""
    so_algebra(θ::SVector, G) -> SMatrix

Build the algebra element `A = θᵃ Gᵃ` (real antisymmetric) from real parameters
`θ` and a generator list `G` (from [`so_generators`](@ref)). `exp(A) ∈ SO(N)`.
"""
so_algebra(θ::SVector, G) = sum(θ[a] * G[a] for a in eachindex(θ))

"""
    so2_exp(A::SMatrix{2,2,Float64}) -> SMatrix{2,2,Float64}

Closed-form SO(2) exponential: a plane rotation by `θ = A[2,1]`.
"""
function so2_exp(A::SMatrix{2,2,Float64})
    θ = A[2, 1]
    return @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

"""
    so3_exp(A::SMatrix{3,3,Float64}) -> SMatrix{3,3,Float64}

Closed-form SO(3) exponential (Rodrigues' rotation formula):
`exp(A) = I + (sin θ / θ) A + ((1 − cos θ)/θ²) A²`, with `θ = √(−½ Tr A²)` the
rotation angle. Faster than the generic `exp`.
"""
function so3_exp(A::SMatrix{3,3,Float64})
    θ2 = -tr(A * A) / 2
    θ  = sqrt(θ2)
    θ < 1e-8 && return one(A) + A + (A * A) / 2          # small-angle series
    return one(A) + (sin(θ) / θ) * A + ((1 - cos(θ)) / θ2) * (A * A)
end

# extend the unified entry point: real (SO) closed forms for 2×2 / 3×3
groupexp(A::SMatrix{2,2,Float64}) = so2_exp(A)
groupexp(A::SMatrix{3,3,Float64}) = so3_exp(A)
