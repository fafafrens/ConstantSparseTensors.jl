# A rank-R tensor that is mostly zero, with a FIXED sparsity pattern. The pattern
# (the nonzero multi-indices) lives in a TYPE parameter, so it is free — no zeros
# stored, no runtime search in contractions. The nonzero VALUES are kept in a tiny
# `SVector` field (general constants like SU(3)'s √3/2 cannot be type parameters;
# only the *pattern* can). So storage = the `K` nonzeros, nothing else.

"""
    ConstantSparseTensor{Dims,IDX,R,K,T}

A rank-`R` tensor of size `Dims` that is mostly zero. The `K` nonzero
multi-indices `IDX` are a type parameter (carried for free, no storage), and the
`K` nonzero values live in an `SVector{K,T}` field. Random `getindex` is an O(K)
unrolled ternary; contractions (`tdot`, [`contract`](@ref)) are `@generated` over
the nonzeros only and allocate nothing.

Build one from a dense array — the pattern is read off the nonzeros:

```julia
ε = ConstantSparseTensor(Float64[(i-j)*(j-k)*(k-i)÷2 for i in 1:3, j in 1:3, k in 1:3])
```
"""
struct ConstantSparseTensor{Dims,IDX,R,K,T}
    vals::SVector{K,T}
end

Base.size(::ConstantSparseTensor{Dims}) where {Dims} = Dims
Base.ndims(::ConstantSparseTensor{Dims,IDX,R}) where {Dims,IDX,R} = R

"""
    nnz(T::ConstantSparseTensor) -> Int

Number of stored (structurally nonzero) entries.
"""
nnz(::ConstantSparseTensor{Dims,IDX,R,K}) where {Dims,IDX,R,K} = K

# build from a dense array (one-time; pattern depends on which entries are nonzero)
function ConstantSparseTensor(A::AbstractArray{T,R}) where {T,R}
    idx = Tuple(Tuple(Iₓ.I) for Iₓ in CartesianIndices(A) if !iszero(A[Iₓ]))
    val = isempty(idx) ? SVector{0,T}() : SVector(Tuple(A[CartesianIndex(ix)] for ix in idx))
    return ConstantSparseTensor{size(A),idx,R,length(idx),T}(val)
end

# getindex: nested ternary over the pattern, unrolled at compile time
@generated function Base.getindex(T::ConstantSparseTensor{Dims,IDX,R,K,Tv},
                                  Iₓ::Vararg{Int,R}) where {Dims,IDX,R,K,Tv}
    expr = :(zero($Tv))
    for k in K:-1:1
        cond = reduce((a, b) -> :($a && $b), (:(Iₓ[$d] == $(IDX[k][d])) for d in 1:R))
        expr = :($cond ? T.vals[$k] : $expr)
    end
    return expr
end

"""
    tdot(T, v₂, …, v_R) -> SVector

Contract slots `2..R` of the rank-`R` tensor `T` with `R-1` vectors, leaving slot
1 free. Recovers the cross product for ε, the identity action for δ, and the Lie
bracket `[p,q]ᶜ` for structure constants `fᵃᵇᶜ`. `@generated`, zero-alloc.
"""
@generated function tdot(T::ConstantSparseTensor{Dims,IDX,R,K,Tv},
                         vs::Vararg{SVector,Nv}) where {Dims,IDX,R,K,Tv,Nv}
    @assert Nv == R - 1 "need R-1 = $(R-1) vectors to leave one slot free, got $Nv"
    m = Dims[1]
    rows = [Any[] for _ in 1:m]
    for k in 1:K
        ix = IDX[k]
        factors = Any[:(T.vals[$k])]
        for s in 2:R
            push!(factors, :(vs[$(s - 1)][$(ix[s])]))
        end
        push!(rows[ix[1]], Expr(:call, :*, factors...))
    end
    comps = map(r -> isempty(r) ? :(zero($Tv)) : Expr(:call, :+, r...), rows)
    return :(SVector{$m,$Tv}($(comps...)))
end

"""
    todense(T::ConstantSparseTensor) -> Array

Materialize back to a plain dense `Array` (verification / interop).
"""
function todense(T::ConstantSparseTensor{Dims,IDX,R,K,Tv}) where {Dims,IDX,R,K,Tv}
    A = zeros(Tv, Dims)
    @inbounds for k in 1:K
        A[CartesianIndex(IDX[k])] = T.vals[k]
    end
    return A
end

"""
    contract(A, B, Val(pairs)) -> ConstantSparseTensor

Contract two `ConstantSparseTensor`s and return another one. `pairs` is a tuple of
`(aSlot, bSlot)` index pairs; each listed slot of `A` is summed against the paired
slot of `B`. The result's slots are the free slots of `A` followed by the free
slots of `B`. The output PATTERN and the grouping of products into output entries
are fixed at compile time from the two patterns; only the value sums
`Σ A.vals[kA]*B.vals[kB]` run at runtime. Zero-alloc.

```julia
casimir = contract(f, f, Val(((1, 1), (2, 2))))   # Σ_{ab} fᵃᵇᵉ fᵃᵇᵍ = N δᵉᵍ
P       = contract(f, f, Val(((3, 1),)))           # P[a,b,c,d] = Σₑ fᵃᵇᵉ fᵉᶜᵈ
```
"""
@generated function contract(A::ConstantSparseTensor{DA,IA,RA,KA,TA},
                             B::ConstantSparseTensor{DB,IB,RB,KB,TB},
                             ::Val{P}) where {DA,IA,RA,KA,TA,DB,IB,RB,KB,TB,P}
    ac    = ntuple(t -> P[t][1], length(P))                 # contracted slots in A
    bc    = ntuple(t -> P[t][2], length(P))                 # contracted slots in B
    afree = Tuple(s for s in 1:RA if !(s in ac))            # surviving slots of A
    bfree = Tuple(s for s in 1:RB if !(s in bc))            # surviving slots of B
    Rc    = length(afree) + length(bfree)
    Tc    = promote_type(TA, TB)
    Dimsc = (ntuple(i -> DA[afree[i]], length(afree))...,
             ntuple(i -> DB[bfree[i]], length(bfree))...)

    contrib = Dict{NTuple{Rc,Int},Vector{Tuple{Int,Int}}}()
    for kA in 1:KA, kB in 1:KB
        all(IA[kA][ac[t]] == IB[kB][bc[t]] for t in eachindex(P)) || continue
        out = (ntuple(i -> IA[kA][afree[i]], length(afree))...,
               ntuple(i -> IB[kB][bfree[i]], length(bfree))...)
        push!(get!(contrib, out, Tuple{Int,Int}[]), (kA, kB))
    end
    outidx = Tuple(sort!(collect(keys(contrib))))
    Kc = length(outidx)
    Kc == 0 && return :(ConstantSparseTensor{$Dimsc,$outidx,$Rc,0,$Tc}(SVector{0,$Tc}()))
    vexpr = map(outidx) do out
        Expr(:call, :+, (:(A.vals[$kA] * B.vals[$kB]) for (kA, kB) in contrib[out])...)
    end
    return :(ConstantSparseTensor{$Dimsc,$outidx,$Rc,$Kc,$Tc}(SVector{$Kc,$Tc}($(vexpr...))))
end
