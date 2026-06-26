# The Levi-Civita tensor ε: mostly zero AND its nonzeros are COMPILE-TIME CONSTANTS
# (±1). Pattern *and* values live entirely in the TYPE, so the object is a SINGLETON
# — zero storage, nothing to index into. (For SU(2), fᵃᵇᶜ = εᵃᵇᶜ.)

"""
    LeviCivita{N}()

The rank-`N` Levi-Civita (permutation) symbol as a zero-storage singleton:
`sizeof(LeviCivita{N}) == 0`. `getindex` is an O(N²) parity formula; the
`tdot` contraction is `@generated` over the `N!` nonzeros with the `±1` folded
into the sign of each term (no multiply by `±1`). `tdot(LeviCivita{3}(), a, b)`
is exactly `cross(a, b)`.
"""
struct LeviCivita{N} end

"""
    lc_sign(idx::NTuple{K,Int}) -> Int

Sign of the permutation `idx` (`+1`/`-1`), or `0` if any index repeats. Pure, so
it is usable inside `@generated` bodies.
"""
@inline function lc_sign(idx::NTuple{K,Int}) where {K}
    s = 1
    @inbounds for a in 1:K, b in (a + 1):K
        idx[a] == idx[b] && return 0
        idx[a] > idx[b] && (s = -s)
    end
    return s
end

@inline Base.getindex(::LeviCivita{N}, I::Vararg{Int,N}) where {N} = lc_sign(I)
Base.size(::LeviCivita{N}) where {N} = ntuple(_ -> N, N)
Base.ndims(::LeviCivita{N}) where {N} = N

# tdot(ε, v₂,…,v_N): contract slots 2..N, leave slot 1 free → SVector{N}. Generated
# over the N! nonzeros, with the permutation sign folded into ± of each term.
# For N = 3 this is the cross product.
"""
    tdot(::LeviCivita{N}, v₂, …, v_N) -> SVector{N}

Contract slots `2..N` of the rank-`N` ε with `N-1` vectors. For `N == 3` this is
the cross product. Generated over the `N!` nonzeros, `±1` folded in, zero-alloc.
"""
@generated function tdot(::LeviCivita{N}, vs::Vararg{SVector{N,T},Nv}) where {N,T,Nv}
    @assert Nv == N - 1 "LeviCivita{$N} needs $(N-1) vectors, got $Nv"
    comps = map(1:N) do i
        terms = Any[]
        for rest in Iterators.product(ntuple(_ -> 1:N, N - 1)...)
            s = lc_sign((i, rest...))
            s == 0 && continue
            prod = Expr(:call, :*, (:(vs[$t][$(rest[t])]) for t in 1:(N - 1))...)
            push!(terms, s == 1 ? prod : :(-$prod))
        end
        isempty(terms) ? :(zero($T)) : Expr(:call, :+, terms...)
    end
    return :(SVector{$N,$T}($(comps...)))
end

"""
    ConstantSparseTensor(::LeviCivita{N})

Materialize the ε singleton as a [`ConstantSparseTensor`](@ref) (e.g. to feed it to
[`contract`](@ref)).
"""
function ConstantSparseTensor(::LeviCivita{N}) where {N}
    A = zeros(Int, ntuple(_ -> N, N))
    for I in CartesianIndices(A)
        A[I] = lc_sign(Tuple(I))
    end
    return ConstantSparseTensor(A)
end
