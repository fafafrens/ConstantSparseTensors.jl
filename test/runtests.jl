using ConstantSparseTensors
using StaticArrays
using LinearAlgebra
using Test

# Zero-allocation check: warm up, then measure inside a function barrier (a bare
# `@allocated f(x)` in @testset scope can over-count via captured locals). The
# calls are allocation-free on Julia ≥ 1.11, which is this package's minimum.
allocs(f, args...) = (f(args...); @allocated f(args...))

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
        @test allocs(tdot, ε, a, b) == 0

        δ = ConstantSparseTensor(Matrix{Float64}(I, 3, 3))
        @test tdot(δ, a) == a
    end

    @testset "contract: ε·ε identity and Casimir" begin
        ε = ConstantSparseTensor(Float64[(i - j) * (j - k) * (k - i) ÷ 2 for i in 1:3, j in 1:3, k in 1:3])
        εε = contract(ε, ε, Val(((1, 1),)))                  # Σᵢ εⁱʲᵏ εⁱˡᵐ
        δ3 = Matrix{Float64}(I, 3, 3)
        ref = [δ3[j, l] * δ3[k, m] - δ3[j, m] * δ3[k, l] for j in 1:3, k in 1:3, l in 1:3, m in 1:3]
        @test todense(εε) ≈ ref
        @test allocs(contract, ε, ε, Val(((1, 1),))) == 0
    end

    @testset "LeviCivita singleton" begin
        ε = LeviCivita{3}()
        @test sizeof(LeviCivita{3}) == 0
        @test ε[1, 2, 3] == 1 && ε[2, 1, 3] == -1 && ε[1, 1, 2] == 0
        a, b = SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)
        @test tdot(ε, a, b) == cross(a, b)
        @test allocs(tdot, ε, a, b) == 0
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
            X = algebra(SVector{N^2 - 1}(0.3 .* randn(N^2 - 1)), T)
            U = groupexp(X)
            @test U ≈ exp(X)           # dispatches to closed form (N=2,3) or exp
            @test U * U' ≈ I(N)        # unitary
            @test det(U) ≈ 1           # special
        end
        # SU(2) closed form (matrix Rodrigues) — fast path + 0 alloc
        X2 = algebra(SVector(0.3, -0.5, 0.8), generators(2))
        @test su2_exp(X2) ≈ exp(X2)
        @test groupexp(X2) ≈ exp(X2)
        @test allocs(su2_exp, X2) == 0
        # SU(3) Morningstar–Peardon closed form — fast path + 0 alloc
        X3 = algebra(SVector{8}(0.3 .* randn(8)), generators(3))
        @test mp_exp(X3) ≈ exp(X3)
        @test groupexp(X3) ≈ exp(X3)
        @test allocs(mp_exp, X3) == 0
    end

    @testset "generic structure_constants + U(N)" begin
        # generic (Hermitian-basis) f matches the dedicated SU(3) one
        f_generic = structure_constants(generators(3))
        f_su3, _  = structure_constants(3)
        @test todense(f_generic) ≈ todense(f_su3)

        # U(N): the u(1) direction commutes ⇒ its f-rows vanish; su(N) block survives
        for N in 2:4
            fu = structure_constants(u_generators(N))
            fud = todense(fu)
            id = N^2                                   # the identity generator's index
            @test all(iszero, fud[id, :, :]) && all(iszero, fud[:, id, :])
            @test maximum(abs, fud .+ permutedims(fud, (2, 1, 3))) < 1e-10   # antisymmetric
        end

        # U(N) exp: trace-safe groupexp lands in U(N) (unitary, det a pure phase)
        for N in 2:4
            X = algebra(SVector{N^2}(0.3 .* randn(N^2)), u_generators(N))
            U = groupexp(X)
            @test U ≈ exp(X)
            @test U * U' ≈ I(N)                        # unitary
            @test abs(det(U)) ≈ 1                      # |det| = 1 (phase, not necessarily 1)
        end
    end

    @testset "Sp / USp(2n) (type Cₙ)" begin
        Ω(n) = symplectic_form(n)
        for n in 1:3
            N = 2n; M = n * (2n + 1)
            G = sp_generators(n)
            @test length(G) == M
            @test all(g -> g ≈ g', G)                  # Hermitian
            @test all(g -> abs(tr(g)) < 1e-12, G)      # traceless
            # structure constants: antisymmetric, Jacobi, Casimir ∝ I
            f = sp_structure_constants(n)
            fd = todense(f)
            @test maximum(abs, fd .+ permutedims(fd, (2, 1, 3))) < 1e-9
            P = todense(contract(f, f, Val(((3, 1),))))
            J = [P[a, b, c, d] + P[b, c, a, d] + P[c, a, b, d]
                 for a in 1:M, b in 1:M, c in 1:M, d in 1:M]
            @test maximum(abs, J) < 1e-9
            C = todense(casimir(f))
            @test C ≈ C[1, 1] * I(M)
            @test C[1, 1] > 0
            # exp lands in USp(2n): unitary AND symplectic UᵀΩU = Ω
            U = groupexp(algebra(SVector{M}(0.3 .* randn(M)), G))
            @test U * U' ≈ I(N)
            @test transpose(U) * Ω(n) * U ≈ Ω(n)
        end
        # usp(2) = su(2)
        @test length(sp_generators(1)) == 3
    end

    @testset "SO(N) structure constants" begin
        for N in 3:5
            M = N * (N - 1) ÷ 2
            f = so_structure_constants(N)
            fd = todense(f)
            @test maximum(abs, fd .+ permutedims(fd, (2, 1, 3))) < 1e-10   # antisymmetric
            # Jacobi
            P = todense(contract(f, f, Val(((3, 1),))))
            J = [P[a, b, c, d] + P[b, c, a, d] + P[c, a, b, d]
                 for a in 1:M, b in 1:M, c in 1:M, d in 1:M]
            @test maximum(abs, J) < 1e-10
            # adjoint Casimir is a positive multiple of the identity
            C = todense(casimir(f))
            @test C ≈ C[1, 1] * I(M)
            @test C[1, 1] > 0
        end
        # so(3) ≅ su(2): six unit-magnitude entries (the ε pattern)
        @test nnz(so_structure_constants(3)) == 6
        f3 = todense(so_structure_constants(3))
        @test sort(abs.(filter(!iszero, vec(f3)))) ≈ ones(6)
        # so(2) abelian
        @test nnz(so_structure_constants(2)) == 0
    end

    @testset "SO(N) exponential map" begin
        for N in 2:5
            M = N * (N - 1) ÷ 2
            A = so_algebra(SVector{M}(0.4 .* randn(M)), so_generators(N))
            R = groupexp(A)
            @test R ≈ exp(A)
            @test R'R ≈ I(N)          # orthogonal
            @test det(R) ≈ 1          # special (SO, not the reflections of O)
        end
        # SO(3) Rodrigues fast path
        A3 = so_algebra(SVector{3}(0.4 .* randn(3)), so_generators(3))
        @test groupexp(A3) === so3_exp(A3)
        @test allocs(so3_exp, A3) == 0
        # SO(2) plane rotation
        A2 = so_algebra(SVector{1}(0.7), so_generators(2))
        @test groupexp(A2) === so2_exp(A2)
        @test allocs(so2_exp, A2) == 0
    end
end
