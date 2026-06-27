"""
    ConstantSparseTensors

Small static tensors that are mostly zero, with the nonzero *pattern* carried in
the type (free — no zeros stored, no runtime search) and the nonzero *values* in a
tiny `SVector`. Contractions are `@generated` over the nonzeros only and allocate
nothing.

Core:
- [`ConstantSparseTensor`](@ref) — pattern-in-type sparse tensor; build from a
  dense array.
- [`tdot`](@ref) — contract slots `2..R` with vectors (cross / identity / Lie
  bracket).
- [`contract`](@ref) — contract two tensors → another `ConstantSparseTensor`
  (pattern fixed at compile time).
- [`todense`](@ref), [`nnz`](@ref).
- [`LeviCivita`](@ref) — the ε symbol as a zero-storage singleton.

Application — SU(N) Lie algebra ([`structure_constants`](@ref), [`generators`](@ref),
[`bracket`](@ref), [`casimir`](@ref), [`adjoint_generators`](@ref)) and the
exponential map ([`algebra`](@ref), [`groupexp`](@ref) — fast closed forms for
SU(2)/SU(3), `exp` fallback otherwise). SO(N) gets the same treatment via
[`so_structure_constants`](@ref), [`so_generators`](@ref), [`so_algebra`](@ref),
and the SO(2)/SO(3) closed forms [`so2_exp`](@ref)/[`so3_exp`](@ref).

The classical groups are complete: Aₙ = SU(N), Bₙ/Dₙ = SO(N), Cₙ = USp(2n)
([`sp_generators`](@ref), [`sp_structure_constants`](@ref)), plus U(N)
([`u_generators`](@ref)). `structure_constants(G)` accepts *any* Hermitian
generator basis, so custom or exceptional groups plug straight in.
"""
module ConstantSparseTensors

using StaticArrays
using LinearAlgebra: tr, det, I, cross, dot, nullspace, eigen, eigvals, norm, Hermitian, kron

export ConstantSparseTensor, nnz, tdot, contract, todense
export LeviCivita, lc_sign
export gellmann, generators, u_generators, structure_constants, structure_constants_dense
export bracket, casimir, jacobi_violation, adjoint_generators
export g2_generators, g2_structure_constants, octonion_structure_constants
export algebra, groupexp, expv, su2_exp, mp_exp
export so_generators, so_structure_constants, so_algebra, so2_exp, so3_exp, so4_exp
export sp_generators, sp_structure_constants, symplectic_form
export quadratic_casimir, dynkin_index, killing_form, adjoint_action
export root_system, RootSystem, weights, highest_weight
export conjugate_rep, tensor_rep, direct_sum_rep, decompose, Irrep, clebsch_gordan
export wigner, invariant_projector, haar_average, haar_sample, trivial_rep, character_projector

include("tensor.jl")
include("levicivita.jl")
include("sun.jl")
include("so.jl")
include("sp.jl")
include("action.jl")
include("invariants.jl")
include("roots.jl")
include("reps.jl")
include("wigner.jl")
include("g2.jl")

end # module
