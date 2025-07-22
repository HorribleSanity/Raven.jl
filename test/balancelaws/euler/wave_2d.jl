using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.Euler

using Test
using Printf
using StaticArrays: SVector
using LinearAlgebra: norm
using MPI
import CUDA
import KernelAbstractions as KA

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV, "RAVEN_INTEGRATION_TESTING", "false")))
end

function wave(law, x⃗, t)
    FT = eltype(law)
    u⃗ = SVector(FT(1), FT(1))
    ρ = 2 + sin(π * (sum(x⃗) - sum(u⃗) * t))
    ρu⃗ = ρ * u⃗
    p = FT(1)
    ρe = Euler.energy(law, ρ, ρu⃗, p)
    SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, K; volume_form = WeakForm(), comm = MPI.COMM_WORLD)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = EulerLaw{FT,2}()

    cell = LobattoCell{FT,A}(Nq, Nq)
    v1d = range(FT(-1), stop = FT(1), length = K + 1)
    coarsegrid = brick((v1d, v1d), (true, true))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RusanovFlux(), comm)

    cfl = FT(1 // 4)
    dt = cfl * step(v1d) / N / Euler.soundspeed(law, FT(1), FT(1))
    timeend = FT(7 // 10)

    q = GridArray(undef, law, grid)
    q .= wave.(Ref(law), points(grid), FT(0))

    normq = weightednorm(dg, q)

    if rank == 0
        @info @sprintf """Starting
        N           = %d
        K           = %d
        volume_form = %s
        norm(q)     = %.16e
        """ N K volume_form normq
    end

    odesolver = LSRK54(dg, q, dt)
    solve!(q, timeend, odesolver)

    qexact = GridArray(undef, law, grid)
    qexact .= wave.(Ref(law), points(grid), timeend)
    errf = weightednorm(dg, q .- qexact)
    normq = weightednorm(dg, q)

    if rank == 0
        @info @sprintf """Finished
        norm(q)      = %.16e
        norm(q - qe) = %.16e
        """ normq errf
    end

    errf
end

let
    if !MPI.Initialized()
        MPI.Init(threadlevel = :multiple)
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    A = Array
    FT = Float64
    N = 4

    if CUDA.functional() && CUDA.has_cuda_gpu()
        CUDA.allowscalar(false)
        A = CUDA.CuArray
    end

    backend = Raven.get_backend(A)

    if backend isa KA.GPU
        local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
        local_rank = MPI.Comm_rank(local_comm)
        KA.device!(backend, (local_rank % KA.ndevices(backend)) + 1)
    end

    if rank == 0
        @info """Using
        A       = $A
        FT      = $FT
        N       = $N
        backend = $backend
        device  = $(KA.device(backend))
       """
    end

    expected_error = Dict()

    #esdg, form
    expected_error[WeakForm(), 1] = 6.1436068743006696e-04
    expected_error[WeakForm(), 2] = 2.5123839524326467e-05
    expected_error[WeakForm(), 3] = 8.6026291789517522e-07

    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 1] =
        1.2210577822045819e-03
    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 2] =
        3.6909108016236636e-05
    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 3] =
        1.4852716431958697e-06

    nlevels = integration_testing ? 3 : 1

    @testset for volume_form in
                 (WeakForm(), FluxDifferencingForm(EntropyConservativeFlux()))
        errors = zeros(FT, nlevels)
        for l = 1:nlevels
            K = 5 * 2^(l - 1)
            errf = run(A, FT, N, K; volume_form, comm)
            errors[l] = errf
            @test errors[l] ≈ expected_error[volume_form, l]
        end

        if nlevels > 1
            rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
            if rank == 0
                @info "Convergence rates\n" * join(
                    ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(nlevels-1)],
                    "\n",
                )
            end
            @test rates[end] ≈ N + 1 atol = 0.4
        end
    end
end
