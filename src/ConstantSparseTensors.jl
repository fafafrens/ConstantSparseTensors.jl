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
SU(2)/SU(3), `exp` fallback otherwise).
"""
module ConstantSparseTensors

using StaticArrays
using LinearAlgebra: tr, det, I, cross

export ConstantSparseTensor, nnz, tdot, contract, todense
export LeviCivita, lc_sign
export gellmann, generators, structure_constants, bracket, casimir, adjoint_generators
export algebra, groupexp, su2_exp, mp_exp

include("tensor.jl")
include("levicivita.jl")
include("sun.jl")

end # module
