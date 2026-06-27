using ConstantSparseTensors
using StaticArrays
using LinearAlgebra
using Random
using Test

# Zero-allocation check: warm up, then measure inside a function barrier (a bare
# `@allocated f(x)` in @testset scope can over-count via captured locals). The calls
# are allocation-free on Julia ≥ 1.12 (stronger escape analysis); on 1.10/1.11 the
# same code allocates a little, so the strict ==0 check is skipped there.
allocs(f, args...) = (f(args...); @allocated f(args...))
const ALLOC_CHECK = VERSION >= v"1.12"

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
        @test allocs(tdot, ε, a, b) == 0 skip = !ALLOC_CHECK

        δ = ConstantSparseTensor(Matrix{Float64}(I, 3, 3))
        @test tdot(δ, a) == a
    end

    @testset "contract: ε·ε identity and Casimir" begin
        ε = ConstantSparseTensor(Float64[(i - j) * (j - k) * (k - i) ÷ 2 for i in 1:3, j in 1:3, k in 1:3])
        εε = contract(ε, ε, Val(((1, 1),)))                  # Σᵢ εⁱʲᵏ εⁱˡᵐ
        δ3 = Matrix{Float64}(I, 3, 3)
        ref = [δ3[j, l] * δ3[k, m] - δ3[j, m] * δ3[k, l] for j in 1:3, k in 1:3, l in 1:3, m in 1:3]
        @test todense(εε) ≈ ref
        @test allocs(contract, ε, ε, Val(((1, 1),))) == 0 skip = !ALLOC_CHECK
    end

    @testset "LeviCivita singleton" begin
        ε = LeviCivita{3}()
        @test sizeof(LeviCivita{3}) == 0
        @test ε[1, 2, 3] == 1 && ε[2, 1, 3] == -1 && ε[1, 1, 2] == 0
        a, b = SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)
        @test tdot(ε, a, b) == cross(a, b)
        @test allocs(tdot, ε, a, b) == 0 skip = !ALLOC_CHECK
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
        @test allocs(su2_exp, X2) == 0 skip = !ALLOC_CHECK
        # SU(3) Morningstar–Peardon closed form — fast path + 0 alloc
        X3 = algebra(SVector{8}(0.3 .* randn(8)), generators(3))
        @test mp_exp(X3) ≈ exp(X3)
        @test groupexp(X3) ≈ exp(X3)
        @test allocs(mp_exp, X3) == 0 skip = !ALLOC_CHECK
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
        @test allocs(so3_exp, A3) == 0 skip = !ALLOC_CHECK
        # SO(2) plane rotation
        A2 = so_algebra(SVector{1}(0.7), so_generators(2))
        @test groupexp(A2) === so2_exp(A2)
        @test allocs(so2_exp, A2) == 0 skip = !ALLOC_CHECK
    end

    @testset "SO(4) closed-form exp" begin
        for _ in 1:5
            A = so_algebra(SVector{6}(0.6 .* randn(6)), so_generators(4))
            R = so4_exp(A)
            @test R ≈ exp(A)
            @test R'R ≈ I(4)
            @test det(R) ≈ 1
        end
        @test groupexp(so_algebra(SVector{6}(0.3 .* randn(6)), so_generators(4))) isa SMatrix{4,4,Float64}
    end

    @testset "expv: action without materializing" begin
        # SO(2)/SO(3): closed-form vector rotation
        A2 = so_algebra(SVector{1}(0.7), so_generators(2)); v2 = SVector(1.0, -2.0)
        @test expv(A2, v2) ≈ groupexp(A2) * v2
        A3 = so_algebra(SVector{3}(0.5 .* randn(3)), so_generators(3)); v3 = SVector(1.0, 2.0, -0.5)
        @test expv(A3, v3) ≈ groupexp(A3) * v3
        # SO(4): cubic-in-A matvec action
        for _ in 1:5
            A4 = so_algebra(SVector{6}(0.6 .* randn(6)), so_generators(4)); v4 = @SVector randn(4)
            @test expv(A4, v4) ≈ exp(A4) * v4
        end
        # SU(2)/SU(3) and U(N): matvec action matches exp(X)*v
        X2 = algebra(SVector(0.3, -0.5, 0.8), generators(2)); w2 = SVector(1.0 + 0im, 2.0 - 1im)
        @test expv(X2, w2) ≈ exp(X2) * w2
        X3 = algebra(SVector{8}(0.4 .* randn(8)), generators(3)); w3 = SVector(1.0 + 0im, -1.0 + 2im, 0.5 + 0im)
        @test expv(X3, w3) ≈ exp(X3) * w3
        Xu = algebra(SVector{4}(0.3 .* randn(4)), u_generators(2))             # u(2): nonzero trace
        @test expv(Xu, w2) ≈ exp(Xu) * w2
        # generic fallback (SU(4))
        X4 = algebra(SVector{15}(0.2 .* randn(15)), generators(4)); w4 = @SVector randn(ComplexF64, 4)
        @test expv(X4, w4) ≈ exp(X4) * w4
        @test allocs(expv, A3, v3) == 0 skip = !ALLOC_CHECK
    end

    @testset "invariants" begin
        for N in 2:4
            @test quadratic_casimir(generators(N)) ≈ ((N^2 - 1) / (2N)) * I(N)   # Schur: ∝ I
            @test dynkin_index(generators(N)) ≈ 0.5                              # T(fund) = ½
        end
        f, _ = structure_constants(3)
        x = SVector{8}(randn(8)); y = SVector{8}(randn(8))
        @test adjoint_action(f, x) * y ≈ bracket(f, x, y)                       # ad_X y = [X,y]
        @test killing_form(f) ≈ 3 * I(8)                                        # = N·I for su(N)
    end

    @testset "root systems (Dynkin from the matrices)" begin
        Random.seed!(0xC0FFEE)
        soH(N) = [SMatrix{N,N,ComplexF64}(im * A) for A in so_generators(N)]    # Hermitian so(N)
        # (generators, rank, #roots, det Cartan)
        cases = [(generators(2), 1, 2, 2), (generators(3), 2, 6, 3), (generators(4), 3, 12, 4),
                 (soH(5), 2, 8, 2), (sp_generators(2), 2, 8, 2), (soH(6), 3, 12, 4)]
        for (G, r, nr, dC) in cases
            rs = root_system(G)
            @test rs.rank == r
            @test length(rs.roots) == nr
            @test length(rs.simple) == r
            @test round(Int, det(rs.cartan)) == dC
            @test all(rs.cartan[i, i] == 2 for i in 1:r)                         # Cartan diagonal
            @test all(rs.cartan[i, j] <= 0 for i in 1:r, j in 1:r if i != j)     # off-diagonal ≤ 0
            @test all((rs.cartan[i, j] == 0) == (rs.cartan[j, i] == 0)           # zeros symmetric
                      for i in 1:r, j in 1:r)
        end
    end

    @testset "weights of a representation" begin
        Random.seed!(0xBEEF)
        # fundamental su(N): N distinct weights, summing to zero, none zero
        for N in 2:4
            ws = weights(generators(N))
            @test length(ws) == N                                       # = dim(fundamental)
            @test norm(sum(ws)) < 1e-8                                  # traceless ⇒ Σ weights = 0
            @test count(w -> norm(w) < 1e-7, ws) == 0                   # no zero weight
            @test length(unique(w -> round.(w, digits = 6), ws)) == N   # all distinct
            @test length(highest_weight(generators(N))) == N - 1        # a rank-vector
        end
        # adjoint rep: weights = roots ∪ {0}^rank
        G = adjoint_generators(3)
        ws = weights(G)
        @test length(ws) == 8
        @test count(w -> norm(w) < 1e-7, ws) == 2                       # zero-weights = rank
        nz = sort(round.(norm.([w for w in ws if norm(w) > 1e-7]), digits = 4))
        @test nz == sort(round.(norm.(root_system(G).roots), digits = 4))  # nonzero weights = roots
    end

    @testset "representations (3, 3̄, tensor products)" begin
        G = generators(3)                                       # the 3
        bar = conjugate_rep(G)                                  # the 3̄
        @test length(bar) == 8 && size(bar[1]) == (3, 3)
        @test all(g -> g ≈ g', bar)                            # Hermitian
        @test todense(structure_constants(bar)) ≈ todense(structure_constants(G))  # same algebra
        @test quadratic_casimir(bar) ≈ (4 / 3) * I(3)          # C₂(3̄) = C₂(3)

        # 3 ⊗ 3̄ = 8 ⊕ 1  (Casimir eigenvalues 3 on the octet, 0 on the singlet)
        T = tensor_rep(G, bar)
        @test length(T) == 8 && size(T[1]) == (9, 9)
        @test all(g -> g ≈ g', T)
        @test sort(real.(eigvals(quadratic_casimir(T)))) ≈ vcat(0.0, fill(3.0, 8)) atol = 1e-7

        # 3 ⊗ 3 = 3̄ ⊕ 6  (C₂(3̄)=4/3, C₂(6)=10/3)
        @test sort(real.(eigvals(quadratic_casimir(tensor_rep(G, G))))) ≈
              vcat(fill(4 / 3, 3), fill(10 / 3, 6)) atol = 1e-7

        # direct sum 3 ⊕ 3̄: block-diagonal, C₂ = (4/3) I₆
        S = direct_sum_rep(G, bar)
        @test size(S[1]) == (6, 6)
        @test quadratic_casimir(S) ≈ (4 / 3) * I(6)
    end

    @testset "irreducible decomposition" begin
        G = generators(3); bar = conjugate_rep(G)
        dm(r) = sort([(ir.dim, ir.multiplicity) for ir in r])
        @test dm(decompose(G)) == [(3, 1)]                              # irreducible
        @test dm(decompose(tensor_rep(G, bar))) == [(1, 1), (8, 1)]     # 3 ⊗ 3̄ = 1 ⊕ 8
        @test dm(decompose(tensor_rep(G, G))) == [(3, 1), (6, 1)]       # 3 ⊗ 3 = 3̄ ⊕ 6
        @test dm(decompose(direct_sum_rep(G, G))) == [(3, 2)]           # 3 ⊕ 3: multiplicity 2
        # 3 and 3̄ are inequivalent but share C₂=4/3 — separated by the commutant center,
        # and NOT merged into one mult-2 irrep (contrast with 3 ⊕ 3 above)
        @test dm(decompose(direct_sum_rep(G, bar))) == [(3, 1), (3, 1)]
        @test dm(decompose(tensor_rep(adjoint_generators(3), G))) ==
              [(3, 1), (6, 1), (15, 1)]                                 # 8 ⊗ 3 = 3 ⊕ 6 ⊕ 15
        # components partition the representation dimension
        d = decompose(tensor_rep(G, bar))
        @test sum(ir.dim * ir.multiplicity for ir in d) == 9
    end

    @testset "Clebsch–Gordan" begin
        A = generators(2)
        U, blocks = clebsch_gordan(A, A)                       # 2 ⊗ 2 = 3 ⊕ 1
        @test U' * U ≈ I(4)                                    # unitary coupling
        @test sort([ir.dim for ir in blocks]) == [1, 3]
        # full block-diagonalization: distinct irrep subspaces don't mix under any generator
        T = tensor_rep(A, A)
        for g in T, a in eachindex(blocks), b in eachindex(blocks)
            a == b && continue
            @test norm(blocks[a].basis' * g * blocks[b].basis) < 1e-8
        end
        # the singlet is (|↑↓⟩ − |↓↑⟩)/√2 up to phase: magnitudes 0, 1/√2, 1/√2, 0
        sing = blocks[findfirst(ir -> ir.dim == 1, blocks)].basis[:, 1]
        @test sort(abs.(sing)) ≈ [0, 0, 1 / √2, 1 / √2] atol = 1e-8
        # SU(3): 3 ⊗ 3̄ couples unitarily into 1 ⊕ 8
        G = generators(3)
        U3, bl3 = clebsch_gordan(G, conjugate_rep(G))
        @test U3' * U3 ≈ I(9)
        @test sort([ir.dim for ir in bl3]) == [1, 8]
    end

    @testset "Wigner D-matrix & Haar average" begin
        Random.seed!(0x5151)
        G = generators(3); θ = randn(8)
        W = wigner(G, θ)
        @test W' * W ≈ I(3)                                 # unitary
        @test W ≈ exp(algebra(SVector{8}(θ), G))            # = ρ(exp(iθ·T))
        # invariant projector: idempotent, Hermitian, rank = trivial multiplicity
        T = tensor_rep(G, conjugate_rep(G))                 # 3 ⊗ 3̄ ⊃ one singlet
        P = invariant_projector(T)
        @test P^2 ≈ P && P' ≈ P
        @test round(Int, real(tr(P))) == 1
        @test invariant_projector(G) ≈ zeros(3, 3)          # the 3 has no invariant
        # ∫_G ρ(g) dg converges to the projector (the averaging theorem)
        @test norm(haar_average(T; samples = 3000) - P) < 0.1
        @test norm(haar_average(G; samples = 3000)) < 0.1   # nontrivial irrep → 0
    end

    @testset "Haar measure & character projectors" begin
        Random.seed!(0x9090)
        G = generators(3)
        V = tensor_rep(G, conjugate_rep(G))                 # 3 ⊗ 3̄ = 1 ⊕ 8
        U = haar_sample(G)
        @test U' * U ≈ I(3)                                 # a group element (unitary)
        # character projector onto the trivial rep = invariant projector (low variance)
        P1 = character_projector(V, trivial_rep(G); samples = 4000)
        @test norm(P1 - invariant_projector(V)) < 0.15
        @test abs(real(tr(P1)) - 1) < 0.3                   # rank 1 (the singlet)
        # onto the octet (R = adjoint): rank 8, and P₁ + P₈ = I
        P8 = character_projector(V, adjoint_generators(3); samples = 4000)
        @test abs(real(tr(P8)) - 8) < 0.6
        @test norm((P1 + P8) - I(9)) < 0.6                  # 1 ⊕ 8 covers everything
        # orthogonality: P_R onto V = R = fundamental is the identity
        @test norm(character_projector(G, G; samples = 4000) - I(3)) < 0.25
    end

    @testset "G₂ (exceptional, dense structure constants)" begin
        Random.seed!(0x6262)
        G = g2_generators()
        @test length(G) == 14 && size(G[1]) == (7, 7)
        @test all(g -> g ≈ g', G)                              # Hermitian
        c = octonion_structure_constants()
        @test maximum(abs, c .+ permutedims(c, (2, 1, 3))) < 1e-12   # octonion 3-form antisym
        f = g2_structure_constants()
        @test f isa Array{Float64,3} && size(f) == (14, 14, 14)      # dense, not sparse
        @test maximum(abs, f .+ permutedims(f, (2, 1, 3))) < 1e-9    # antisymmetric
        @test jacobi_violation(f) < 1e-9                            # Jacobi (dense loops)
        C = casimir(f)
        @test C ≈ C[1, 1] * I(14) && C[1, 1] > 0                    # Casimir ∝ I
        x = randn(14); y = randn(14)
        @test bracket(f, x, y) ≈ -bracket(f, y, x)                  # dense bracket, antisymmetric
        # root system recovers G₂: rank 2, 12 roots, det Cartan = 1 (the triple bond)
        rs = root_system(G)
        @test rs.rank == 2 && length(rs.roots) == 12 && round(Int, det(rs.cartan)) == 1
        # the 7-dimensional representation is irreducible
        @test [(ir.dim, ir.multiplicity) for ir in decompose(G)] == [(7, 1)]
        # exp lands in SO(7) (G₂ ⊂ SO(7))
        U = groupexp(algebra(SVector{14}(0.3 .* randn(14)), G))
        @test U * U' ≈ I(7) && real(det(U)) ≈ 1
    end

    @testset "E₆ (Chevalley construction)" begin
        Random.seed!(0x00E6)
        A = e6_cartan_matrix()
        @test round(Int, det(A)) == 3                       # E₆ Cartan determinant
        roots = e6_roots()
        @test length(roots) == 72                           # E₆ root count
        @test all(dot(α, A * α) == 2 for α in roots)        # all roots length² = 2
        C = e6_structure_constants()
        @test size(C) == (78, 78, 78)                       # dim E₆ = 78
        @test maximum(abs(C[a, b, c] + C[b, a, c]) for a in 1:78, b in 1:78, c in 1:78) == 0  # antisym
        # Jacobi (sampled — the full check is 78⁵): certifies the sign cocycle
        D = 78; viol = 0.0
        for _ in 1:5000
            a, b, c, e = rand(1:D, 4)
            s = sum(C[a, b, d] * C[d, c, e] + C[b, c, d] * C[d, a, e] + C[c, a, d] * C[d, b, e] for d in 1:D)
            viol = max(viol, abs(s))
        end
        @test viol < 1e-10
        # the generic builder gives the right dimension for other simply-laced types
        @test size(first(chevalley_structure_constants([2 -1; -1 2])), 1) == 8           # A₂ = su(3)
        @test size(first(chevalley_structure_constants(
                  [2 -1 0 0; -1 2 -1 -1; 0 -1 2 0; 0 -1 0 2])), 1) == 28                  # D₄
        # compact Hermitian adjoint generators → root_system recovers the E₆ Cartan
        G = e6_generators()
        @test length(G) == 78 && size(G[1]) == (78, 78)
        @test all(g -> maximum(abs, g - g') < 1e-9, G)                       # Hermitian
        rs = root_system(G)
        @test rs.rank == 6 && length(rs.roots) == 72 && round(Int, det(rs.cartan)) == 3
    end
end
