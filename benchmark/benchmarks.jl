# Benchmarks for ConstantSparseTensors.jl — reproduces the headline numbers in the
# README (sparse vs dense contraction, closed-form vs generic exp, expv vs forming
# the matrix). Numbers are median wall time; absolute values are machine-dependent.
#
# Setup (once), from the package root:
#     julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
# Run:
#     julia --project=benchmark benchmark/benchmarks.jl

using ConstantSparseTensors, StaticArrays, LinearAlgebra, BenchmarkTools, Printf

ns(t) = @sprintf("%8.1f ns", t * 1e9)

function main()
    println("== ConstantSparseTensors benchmarks (median wall time) ==\n")

    # 1. SU(3) structure-constant contraction: sparse tdot vs the dense triple loop
    f, _ = structure_constants(3); fd = todense(f)
    p, q = (@SVector randn(8)), (@SVector randn(8))
    ts = @belapsed tdot($f, $p, $q)
    td = @belapsed SVector{8}(sum($fd[i, j, k] * $p[j] * $q[k] for j in 1:8, k in 1:8) for i in 1:8)
    @printf("SU(3) f-contraction    sparse %s   dense %s   (%.0f×)\n", ns(ts), ns(td), td / ts)
    @printf("SU(3) Casimir (sparse×sparse contract)    %s\n", ns(@belapsed casimir($f)))

    # 2. exponential map: group-specific closed forms vs the generic `exp`
    println()
    X2 = algebra(SVector(0.3, -0.5, 0.8), generators(2))
    X3 = algebra(SVector{8}(0.3 .* randn(8)), generators(3))
    A3 = so_algebra(SVector{3}(0.4 .* randn(3)), so_generators(3))
    A4 = so_algebra(SVector{6}(0.4 .* randn(6)), so_generators(4))
    @printf("SU(2) exp    su2_exp %s   generic exp %s\n", ns(@belapsed su2_exp($X2)), ns(@belapsed exp($X2)))
    @printf("SU(3) exp     mp_exp %s   generic exp %s\n", ns(@belapsed mp_exp($X3)),  ns(@belapsed exp($X3)))
    @printf("SO(3) exp    so3_exp %s   generic exp %s\n", ns(@belapsed so3_exp($A3)), ns(@belapsed exp($A3)))
    @printf("SO(4) exp    so4_exp %s   generic exp %s\n", ns(@belapsed so4_exp($A4)), ns(@belapsed exp($A4)))

    # 3. expv (action on a vector) vs forming exp(X) then multiplying
    println()
    v3 = @SVector randn(3); w3 = @SVector randn(ComplexF64, 3); v4 = @SVector randn(4)
    @printf("SO(3) action   expv %s   groupexp*v %s\n", ns(@belapsed expv($A3, $v3)), ns(@belapsed groupexp($A3) * $v3))
    @printf("SU(3) action   expv %s   groupexp*v %s\n", ns(@belapsed expv($X3, $w3)), ns(@belapsed groupexp($X3) * $w3))
    @printf("SO(4) action   expv %s   groupexp*v %s\n", ns(@belapsed expv($A4, $v4)), ns(@belapsed groupexp($A4) * $v4))
    return nothing
end

main()
