using ConstantSparseTensors
using StaticArrays
using LinearAlgebra
using Test

@testset "ConstantSparseTensors.jl" begin

    @testset "ConstantSparseTensor: build / getindex / todense" begin
        εd = [(i - j) * (j - k) * (k - i) ÷ 2 for i in 1:3, j in 1:3, k in 1:3]
        ε  = ConstantSparseTensor(Float64.(εd))
        @test nnz(ε) == 6
        @test size(ε) == (3, 3, 3)
        @test ndims(ε) == 3
        @test ε[1, 2, 3] == 1.0
        @test ε[2, 1, 3] == -1.0
        @test ε[1, 1, 2] == 0.0
        @test todense(ε) == Float64.(εd)
    end

    @testset "tdot: cross / identity / bracket" begin
        ε = ConstantSparseTensor(Float64[(i - j) * (j - k) * (k - i) ÷ 2 for i in 1:3, j in 1:3, k in 1:3])
        a, b = SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)
        @test tdot(ε, a, b) == cross(a, b)
        @test (@allocated tdot(ε, a, b)) == 0

        δ = ConstantSparseTensor(Matrix{Float64}(I, 3, 3))
        @test tdot(δ, a) == a
    end

    @testset "contract: ε·ε identity and Casimir" begin
        ε = ConstantSparseTensor(Float64[(i - j) * (j - k) * (k - i) ÷ 2 for i in 1:3, j in 1:3, k in 1:3])
        εε = contract(ε, ε, Val(((1, 1),)))                  # Σᵢ εⁱʲᵏ εⁱˡᵐ
        δ3 = Matrix{Float64}(I, 3, 3)
        ref = [δ3[j, l] * δ3[k, m] - δ3[j, m] * δ3[k, l] for j in 1:3, k in 1:3, l in 1:3, m in 1:3]
        @test todense(εε) ≈ ref
        @test (@allocated contract(ε, ε, Val(((1, 1),)))) == 0
    end

    @testset "LeviCivita singleton" begin
        ε = LeviCivita{3}()
        @test sizeof(LeviCivita{3}) == 0
        @test ε[1, 2, 3] == 1 && ε[2, 1, 3] == -1 && ε[1, 1, 2] == 0
        a, b = SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)
        @test tdot(ε, a, b) == cross(a, b)
        @test (@allocated tdot(ε, a, b)) == 0
        # materialize and cross-check against the built tensor
        @test todense(ConstantSparseTensor(ε)) == [lc_sign((i, j, k)) for i in 1:3, j in 1:3, k in 1:3]
    end

    @testset "SU(N) structure constants (N = 2..5)" begin
        for N in 2:5
            M = N^2 - 1
            f, d = structure_constants(N)
            fd, dd = todense(f), todense(d)
            @test maximum(abs, fd .+ permutedims(fd, (2, 1, 3))) < 1e-10   # f antisymmetric
            @test maximum(abs, dd .- permutedims(dd, (2, 1, 3))) < 1e-10   # d symmetric
            @test todense(casimir(f)) ≈ N * I(M)                          # Σ fᵃᵇᵉfᵃᵇᵍ = N δ
        end
        # SU(2): d ≡ 0, f = ε
        f2, d2 = structure_constants(2)
        @test nnz(d2) == 0
        @test todense(f2) ≈ [lc_sign((i, j, k)) for i in 1:3, j in 1:3, k in 1:3]
    end

    @testset "SU(3) bracket / Jacobi / adjoint" begin
        N, M = 3, 8
        f, _ = structure_constants(N)
        p, q = (@SVector randn(M)), (@SVector randn(M))
        fd = todense(f)
        dense = SVector{M}(sum(fd[i, j, k] * p[j] * q[k] for j in 1:M, k in 1:M) for i in 1:M)
        @test bracket(f, p, q) ≈ dense
        @test bracket(f, p, q) ≈ -bracket(f, q, p)             # antisymmetry

        # Jacobi  Σ_cyclic fᵃᵇᵉfᵉᶜᵈ = 0
        P = todense(contract(f, f, Val(((3, 1),))))
        J = [P[a, b, c, d] + P[b, c, a, d] + P[c, a, b, d]
             for a in 1:M, b in 1:M, c in 1:M, d in 1:M]
        @test maximum(abs, J) < 1e-12

        # adjoint generators close the algebra and have Casimir N
        Tadj = adjoint_generators(N)
        a0, b0 = 1, 2
        comm = Tadj[a0] * Tadj[b0] - Tadj[b0] * Tadj[a0]
        @test maximum(abs, comm - sum(im * fd[a0, b0, c] * Tadj[c] for c in 1:M)) < 1e-10
        K = [real(tr(Tadj[a] * Tadj[b])) for a in 1:M, b in 1:M]
        @test K ≈ N * I(M)
    end

    @testset "exponential map: SU(N)" begin
        for N in 2:6
            T = generators(N)
            θ = SVector{N^2 - 1}(0.3 .* randn(N^2 - 1))
            X = algebra(θ, T)
            U = expch(X)
            @test U ≈ exp(X)
            @test U * U' ≈ I(N)        # unitary
            @test det(U) ≈ 1           # special
            @test (@allocated expch(X)) == 0
        end
        # SU(2) Rodrigues closed form
        θ = SVector(0.3, -0.5, 0.8)
        @test su2_exp(θ) ≈ exp(algebra(θ, generators(2)))
        # SU(3) Morningstar–Peardon closed form
        X3 = algebra(SVector{8}(0.3 .* randn(8)), generators(3))
        @test mp_exp(X3) ≈ exp(X3)
        @test (@allocated mp_exp(X3)) == 0
    end
end
