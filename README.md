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
| G₂ | g₂ = Aut(𝕆) | `g2_structure_constants()` / `g2_generators()` |
| — | u(N) | `u_generators(N)` |

**G₂** is built from the octonions (Cayley–Dickson) as the derivation algebra inside
so(7), and `root_system(g2_generators())` recovers its Cartan matrix `[[2,-1],[-3,2]]`
— the exceptional triple bond — straight from octonion multiplication. Its structure
constants are kept **dense** (`g2_structure_constants() :: Array`): at 14 dimensions
the `@generated` `contract` is impractical (compile cost ~`nnz²`), so the dense
`casimir`/`bracket`/`jacobi_violation` methods are used. The engine stays for the
*small* constant tensors it's good at.

**E₆** (dim 78) is built by the **Chevalley construction** — `chevalley_structure_constants(A)`
takes any simply-laced Cartan matrix, generates its root system, fixes the
structure-constant signs with a 2-cocycle on the root lattice, and assembles `f` in
the Cartan–Weyl basis. The result satisfies Jacobi (`jacobi_violation ≈ 0`), the
oracle for the signs:

```julia
A = e6_cartan_matrix()              # det 3, the E₆ Cartan matrix
length(e6_roots())                  # 72 roots, all length² = 2
C = e6_structure_constants()        # 78×78×78, jacobi_violation(C) ≈ 0
chevalley_structure_constants(A)    # works for any simply-laced A (Aₙ, Dₙ, E₆/₇/₈)

G = e6_generators()                 # 78 compact Hermitian adjoint generators
root_system(G).cartan               # recovers the E₆ Cartan (rank 6, 72 roots, det 3)
```

`e6_generators()` passes the split Cartan–Weyl form to the **compact real form**
`{i hⱼ, e_α+e_{-α}, i(e_α−e_{-α})}` and orthonormalizes under the Killing form, giving
Hermitian adjoint generators — so the whole toolkit (`root_system`, `casimir`,
`weights`) applies, and `root_system` rediscovers E₆ from the matrices. Same builder
gives E₇/E₈ (and the classical Aₙ/Dₙ) — only the Cartan matrix changes.

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

### Representations

The fundamental is `generators(N)`; build the others from it, then feed them to any
of the tools above:

```julia
G   = generators(3)          # the 3
bar = conjugate_rep(G)       # the 3̄  (−conj(Tᵃ); same Casimir, negated weights)

T = tensor_rep(G, bar)       # 3 ⊗ 3̄  (Aᵃ⊗I + I⊗Bᵃ, dim 9)
eigvals(quadratic_casimir(T))   # → 3 (×8) and 0 (×1):  3 ⊗ 3̄ = 8 ⊕ 1

tensor_rep(G, G)             # 3 ⊗ 3 = 3̄ ⊕ 6   (Casimir 4/3 on the 3̄, 10/3 on the 6)
direct_sum_rep(G, bar)       # 3 ⊕ 3̄  (block diagonal)

# decompose a (reducible) representation into irreducibles
decompose(tensor_rep(G, bar))        # 1 ⊕ 8
decompose(tensor_rep(G, G))          # 3 ⊕ 6   (the 3̄ and the 6)
decompose(direct_sum_rep(G, G))      # one irrep of dim 3, multiplicity 2
decompose(direct_sum_rep(G, bar))    # 3 and 3̄: two inequivalent dim-3 irreps
decompose(tensor_rep(adjoint_generators(3), G))   # 8 ⊗ 3 = 3 ⊕ 6 ⊕ 15
```

`decompose` works via the **commutant** `𝒞 = {M : [M,Tᵃ]=0} = ⊕ᵢ M_{mᵢ}(ℂ)`: a
generic element of its center separates the isotypic components (distinguishing
*all* inequivalent irreps — even a rep and its conjugate, which share a Casimir),
and Schur's lemma (`dim 𝒞ᵢ = mᵢ²`) gives each **multiplicity**. So `3 ⊕ 3` is the
**3** with multiplicity 2, while `3 ⊕ 3̄` is correctly two distinct dim-3 irreps.
`weights`, `structure_constants`, `root_system` all work on any of these
representations.

### Clebsch–Gordan

The invariant-subspace bases from `decompose` are the **Clebsch–Gordan coefficients**
— the unitary coupling the product into irreducibles:

```julia
U, blocks = clebsch_gordan(generators(2), generators(2))   # 2 ⊗ 2 = 3 ⊕ 1
# U block-diagonalizes every Aᵃ⊗I + I⊗Bᵃ; columns = coupled states in the product basis.
# the singlet column is (|↑↓⟩ − |↓↑⟩)/√2  →  magnitudes 0, 1/√2, 1/√2, 0
```

Determined up to the unavoidable unitary freedom inside each irrep (phase /
multiplicity / weight-basis convention); the convention-independent quantities (the
irrep content, multiplicities, and e.g. the singlet's `1/√2`) are exact.

### Wigner D-matrices & the Haar average

`wigner(G, θ)` is the representation of a group element `exp(i θᵃ Tᵃ)` — i.e. `ρ(g)`.
Averaging it over the group (Haar measure) gives the **projector onto the invariant
subspace** (`∫_G ρ(g) dg = ∩ₐ ker Tᵃ`):

```julia
W = wigner(generators(3), randn(8))          # a 3×3 SU(3) Wigner matrix (unitary)

T = tensor_rep(generators(3), conjugate_rep(generators(3)))   # 3 ⊗ 3̄
P = invariant_projector(T)                   # projector onto the singlet, rank 1, P² = P
haar_average(T) ≈ P                          # Monte-Carlo ∫ρ(g)dg → the projector
invariant_projector(generators(3))           # 0: the 3 has no invariant
```

`invariant_projector` is exact (the common kernel of the generators); `haar_average`
is the Monte-Carlo group average that converges to it — the averaging theorem made
into a test.

Exposing the measure (`haar_sample`) gives projectors onto **any** irrep via the
**character projection** `P_R = d_R ∫_G conj(χ_R(g)) ρ_V(g) dg` (Peter–Weyl):

```julia
V = tensor_rep(generators(3), conjugate_rep(generators(3)))   # 3 ⊗ 3̄ = 1 ⊕ 8
character_projector(V, adjoint_generators(3))   # projector onto the octet (rank 8)
character_projector(V, trivial_rep(generators(3)))  # the singlet  (= invariant_projector)
# the two sum to the identity:  P₈ + P₁ = I₉
```

`character_projector(V, R)` weights the Haar average by the character of `R` — so the
invariant projector is just the `R = `trivial case (`χ ≡ 1`).

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
