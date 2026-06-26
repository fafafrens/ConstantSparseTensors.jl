# ConstantSparseTensors.jl

Small static tensors that are **mostly zero**, with the nonzero *pattern* carried in
the **type** (free — no zeros stored, no runtime search) and the nonzero *values* in
a tiny `SVector`. Contractions are `@generated` over the nonzeros only and allocate
nothing. Includes an SU(N) Lie-algebra application built entirely on the engine.

Depends only on `StaticArrays` (+ stdlib `LinearAlgebra`).

## The idea

A constant sparse tensor stores its nonzero multi-indices as a type parameter and
its `K` values in an `SVector{K}`:

```julia
struct ConstantSparseTensor{Dims,IDX,R,K,T}
    vals::SVector{K,T}
end
```

Because the pattern is a compile-time constant, every contraction can decide *which
products land where* during compilation and emit straight-line, zero-allocation
arithmetic — the runtime never searches indices or stores a zero.

## Engine

```julia
using ConstantSparseTensors, StaticArrays

# build from a dense array — the pattern is read off the nonzeros
ε = ConstantSparseTensor(Float64[(i-j)*(j-k)*(k-i)÷2 for i in 1:3, j in 1:3, k in 1:3])

a, b = SVector(1.0,2.0,3.0), SVector(4.0,5.0,6.0)
tdot(ε, a, b) == cross(a, b)            # contract slots 2..R with vectors  (0 alloc)

# contract two tensors → ANOTHER ConstantSparseTensor (pattern fixed at compile time)
εε = contract(ε, ε, Val(((1,1),)))      # Σᵢ εⁱʲᵏ εⁱˡᵐ = δⱼˡδₖₘ − δⱼₘδₖₗ
```

- `ConstantSparseTensor(A)` — build from a dense array.
- `tdot(T, v₂,…,v_R)` — contract slots `2..R` with vectors, leave slot 1 free
  (cross product for ε, identity for δ, Lie bracket for `fᵃᵇᶜ`).
- `contract(A, B, Val(pairs))` — contract slot pairs of two tensors → another
  `ConstantSparseTensor`; only the value sums run at runtime.
- `todense(T)`, `nnz(T)`.
- `LeviCivita{N}()` — the ε symbol as a **zero-storage singleton**
  (`sizeof == 0`); `tdot` lowers to the bare `±` expression.

## Application: SU(N)

The structure constants and everything they generate, for arbitrary `N`:

```julia
f, d = structure_constants(3)           # fᵃᵇᶜ (antisym), dᵃᵇᶜ (sym) as sparse tensors

bracket(f, p, q)                        # Lie bracket on coefficient vectors = tdot(f,p,q)
casimir(f)                              # Σ_{ab} fᵃᵇᵉfᵃᵇᵍ = N δ   (via `contract`)
adjoint_generators(3)                   # (Tᵃ)_bc = −i fᵃᵇᶜ

# exponential map  algebra → group  (lattice gauge-link / HMC update)
T = generators(3)
X = algebra(SVector{8}(0.3 .* randn(8)), T)   # X = i θᵃTᵃ, anti-Hermitian
U = groupexp(X)                         # fast closed form for SU(2)/SU(3), `exp` otherwise
mp_exp(X)                               # Morningstar–Peardon SU(3) closed form  (~1.4× exp)
su2_exp(X2)                             # SU(2) matrix Rodrigues closed form
```

### SO(N) / O(N)

`so(N)` (real antisymmetric matrices) gets the same treatment — same engine, same
`bracket`/`casimir`:

```julia
f = so_structure_constants(4)           # so(N) fᵃᵇᶜ, dimension N(N−1)/2; so(3) = ε
G = so_generators(3)
A = so_algebra(SVector(0.4, -0.2, 0.5), G)    # real antisymmetric
R = groupexp(A)                         # SO(3) Rodrigues; ∈ SO(N), RᵀR = I, det = 1
so2_exp(A2)                             # SO(2) plane rotation, closed form
```

`exp` of the algebra lands in **SO(N)** (det +1); `O(N)` adds the det −1
reflections, which `exp` cannot reach. `groupexp` routes real 2×2/3×3 to the
rotation closed forms and complex 2×2/3×3 to `su2_exp`/`mp_exp`, falling back to
`exp` for everything else.

On the exponential: for a generic static matrix, **just use `exp`** —
`StaticArrays` already exponentiates `SMatrix` allocation-free, and a hand-rolled
Cayley–Hamilton series was benchmarked to be *slower* for `N ≠ 2, 3`. The only
wins are the group-specific closed forms `su2_exp` / `mp_exp`, which exploit the
degree-2 minimal polynomial. `groupexp` picks the fast path automatically and
falls back to `exp`.

`f`/`d` stay sparse as `N` grows (only a few % of `(N²−1)³` entries are nonzero), so
the SU(3) bracket runs ~250× faster than the dense `8×8×8` contraction. SU(2): `f`
is the Levi-Civita ε and `d ≡ 0`.

## Install / test

```julia
using Pkg
Pkg.develop(path="path/to/ConstantSparseTensors.jl")
Pkg.test("ConstantSparseTensors")
```

## Background

Grew out of finite-volume / lattice gauge-theory needs: small constant coefficient
tensors (ε, δ, structure constants) where storing zeros and searching indices at
runtime is pure overhead. The Morningstar–Peardon SU(3) form follows
[Morningstar & Peardon, *Phys. Rev. D* **69**, 054501 (2004)].
