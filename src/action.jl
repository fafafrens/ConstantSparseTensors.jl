# expv(X, v) = exp(X)·v — apply a group element to a vector WITHOUT materializing
# exp(X). For the closed-form cases this is a direct formula: SO(3) is the Rodrigues
# vector rotation; SU(2)/SU(3) reuse the Cayley–Hamilton coefficients but apply the
# generator to v by matvec (X·v, Q·v) instead of forming the full exponential.

"""
    expv(X, v) -> SVector

Apply the group element `exp(X)` to the vector `v` without forming `exp(X)`.
Closed-form for SO(2)/SO(3) (plane / Rodrigues rotation) and SU(2)/SU(3) (the
algebra acts on `v` by matvec); falls back to `groupexp(X) * v` otherwise.
"""
expv(X::SMatrix, v::SVector) = groupexp(X) * v

# SO(2): rotate v by θ = A[2,1]
function expv(A::SMatrix{2,2,Float64}, v::SVector{2})
    θ = A[2, 1]; c = cos(θ); s = sin(θ)
    return SVector(c * v[1] - s * v[2], s * v[1] + c * v[2])
end

# SO(3): Rodrigues' rotation of v about axis ω = vee(A), angle |ω| — no matrix.
function expv(A::SMatrix{3,3,Float64}, v::SVector{3})
    ω  = SVector(A[3, 2], A[1, 3], A[2, 1])      # A·x = ω × x
    θ2 = ω[1]^2 + ω[2]^2 + ω[3]^2
    θ  = sqrt(θ2)
    θ < 1e-8 && return v + cross(ω, v)           # small-angle: exp(A) ≈ I + A
    k = ω / θ
    return cos(θ) * v + sin(θ) * cross(k, v) + (1 - cos(θ)) * dot(k, v) * k
end

# SU(2)/U(2): exp(X)v = e^τ (cos λ · v + (sin λ/λ) X₀·v), X₀ the traceless part.
function expv(X::SMatrix{2,2,ComplexF64}, v::SVector{2})
    τ  = tr(X) / 2; X0 = X - τ * one(X)
    λ2 = max(real(det(X0)), 0.0)                 # X₀² = −det(X₀) I, eigenvalues ±iλ
    λ  = sqrt(λ2)
    base = λ < 1e-12 ? v + X0 * v : cos(λ) * v + (sin(λ) / λ) * (X0 * v)
    return exp(τ) * base
end

# SU(3)/U(3): exp(X)v = e^τ (f₀ v + f₁ Q·v + f₂ Q·(Q·v)), Q = −i X₀ — two matvecs.
function expv(X::SMatrix{3,3,ComplexF64}, v::SVector{3})
    τ = tr(X) / 3; X0 = X - τ * one(X)
    Q = -im * X0
    f0, f1, f2 = _mp_coeffs(Q)
    Qv = Q * v
    return exp(τ) * (f0 * v + f1 * Qv + f2 * (Q * Qv))
end
