# ConstantSparseTensors.jl

[![CI](https://github.com/fafafrens/ConstantSparseTensors.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/fafafrens/ConstantSparseTensors.jl/actions/workflows/CI.yml)

Small static tensors that are **mostly zero**, with the nonzero *pattern* carried in
the **type** (free — no zeros stored, no runtime search) and the nonzero *values* in
a tiny `SVector`. Contractions are `@generated` over the nonzeros only and allocate
nothing. Includes a classical-group (SU/SO/Sp/U) Lie-algebra application built
entirely on the engine.

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

expv(X, v)                              # exp(X)·v WITHOUT forming exp(X)
                                        #   SO(3): Rodrigues vector rotation; SU(N): matvec action
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
so4_exp(A4)                             # SO(4) closed form (so(4)=su(2)⊕su(2))
```

`exp` of the algebra lands in **SO(N)** (det +1); `O(N)` adds the det −1
reflections, which `exp` cannot reach. `groupexp` routes real 2×2/3×3/4×4 to the
rotation closed forms and complex 2×2/3×3 to `su2_exp`/`mp_exp`, falling back to
`exp` for everything else.

### All the classical groups + a generic API

The classical families are complete, and one generic primitive does all the work:

| Cartan type | algebra | constructor |
|---|---|---|
| Aₙ | su(N) | `structure_constants(N)` / `generators(N)` |
| Bₙ, Dₙ | so(N) | `so_structure_constants(N)` / `so_generators(N)` |
| Cₙ | usp(2n) | `sp_structure_constants(n)` / `sp_generators(n)` |
| — | u(N) | `u_generators(N)` |

```julia
f = sp_structure_constants(2)            # USp(4) structure constants (dim 10)
G = sp_generators(2)
U = groupexp(algebra(SVector{10}(0.2 .* randn(10)), G))   # ∈ USp(4): UᵀΩU = Ω

structure_constants(my_generators)       # ANY orthogonal Hermitian basis
```

`structure_constants(G)` takes any orthonormal **Hermitian** generator basis and
returns `fᵃᵇᶜ` as a `ConstantSparseTensor` — so custom or exceptional groups plug
in without touching the engine. `bracket`, `casimir`, `adjoint_generators`,
`contract` (Jacobi) then all work as before.

### Invariants & the root system

The standard Lie-algebra quantities fall out of the generators and `f`:

```julia
quadratic_casimir(generators(3))         # C₂ = Σ TᵃTᵃ = (N²−1)/2N · I   (Schur)
dynkin_index(generators(3))              # T(R): Tr(TᵃTᵇ) = T(R) δ        (½ for fund)
killing_form(f)                          # κᵃᵇ = fᵃᶜᵈfᵇᶜᵈ  (∝ I, simple algebra)
adjoint_action(f, x)                     # matrix of ad_X = [X,·];  exp(·) = Ad

# recover the Dynkin type from the matrices alone
rs = root_system(generators(3))          # su(3) = A₂
rs.rank          # 2
length(rs.roots) # 6
rs.cartan        # [2 -1; -1 2]

# weights of a representation (simultaneous Cartan eigenvalues)
weights(generators(3))                   # 3 weights of the fundamental
highest_weight(generators(3))
weights(adjoint_generators(3))           # 8 weights = the 6 roots + 2 zero-weights
```

`root_system` finds a Cartan subalgebra (centralizer of a generic element),
reads the **roots** off the adjoint eigenvectors, and returns the **simple roots**
and **Cartan matrix** — rediscovering the classification (su(N)=Aₙ, so=Bₙ/Dₙ,
sp=Cₙ) straight from the generator matrices. `weights(G)` diagonalizes the Cartan
subalgebra in the representation `G`; for the adjoint rep the nonzero weights are
exactly the roots.

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
Pkg.add(url="https://github.com/fafafrens/ConstantSparseTensors.jl")
```

Requires Julia 1.10+. (The contraction/exp kernels are allocation-free on Julia
1.12+; on 1.10/1.11 they allocate a little, which is why the strict zero-alloc tests
are gated to 1.12.)

Local development:

```julia
Pkg.develop(path="path/to/ConstantSparseTensors.jl")
Pkg.test("ConstantSparseTensors")
```

## Benchmarks

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/benchmarks.jl
```

reproduces the headline numbers (on an M-series CPU):

```
SU(3) f-contraction    sparse  17.7 ns   dense  927 ns   (52×)
SU(2) exp    su2_exp  31 ns   generic exp 106 ns          SO(3) exp  so3_exp 28 ns  exp  87 ns
SU(3) exp     mp_exp 251 ns   generic exp 355 ns          SO(4) exp  so4_exp 105 ns exp 211 ns
```

## Background

Grew out of finite-volume / lattice gauge-theory needs: small constant coefficient
tensors (ε, δ, structure constants) where storing zeros and searching indices at
runtime is pure overhead. The Morningstar–Peardon SU(3) form follows
[Morningstar & Peardon, *Phys. Rev. D* **69**, 054501 (2004)].
