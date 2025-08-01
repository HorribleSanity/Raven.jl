using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.Advection

using Test
using Printf
using LinearAlgebra: norm
using StaticArrays: SVector
using MPI
import CUDA
import KernelAbstractions as KA

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV, "RAVEN_INTEGRATION_TESTING", "false")))
end

function wave(law, x⃗, t)
    ρ = 2 + sin(π * (sum(x⃗) - sum(constants(law).u⃗) * t))
    SVector(ρ)
end

function run(A, FT, N, K; volume_form = WeakForm(), comm = MPI.COMM_WORLD)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = AdvectionLaw{FT,1}()

    cell = LobattoCell{FT,A}(Nq)
    v1d = range(FT(-1), stop = FT(1), length = K + 1)
    coarsegrid = brick((v1d,), (true,))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RusanovFlux(), comm)

    cfl = FT(1 // 2)
    dt = cfl * step(v1d) / N / norm(constants(law).u⃗)
    timeend = FT(0.7)

    q = GridArray(undef, law, grid)
    q .= wave.(Ref(law), points(grid), FT(0))

    normq = weightednorm(dg, q)
    if rank == 0
        @info @sprintf """Starting
        N                   = %d
        K                   = %d
        FT                  = %s
        A                   = %s
        backend             = %s
        volume_form         = %s
        integration_testing = %s
        norm(q)             = %.16e
        """ N K FT A KA.get_backend(q) volume_form integration_testing normq
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

    expected_error = Dict()

    #form, lev
    expected_error[WeakForm(), 1] = 1.5238393162339734e-04
    expected_error[WeakForm(), 2] = 4.8377348010293410e-06
    expected_error[WeakForm(), 3] = 1.4063471307765498e-07
    expected_error[WeakForm(), 4] = 4.5781066929825930e-09

    expected_error[FluxDifferencingForm(CentralFlux()), 1] = 1.5238393162391597e-04
    expected_error[FluxDifferencingForm(CentralFlux()), 2] = 4.8377348011318472e-06
    expected_error[FluxDifferencingForm(CentralFlux()), 3] = 1.4063471309426943e-07
    expected_error[FluxDifferencingForm(CentralFlux()), 4] = 4.5781065356312460e-09

    nlevels = integration_testing ? 4 : 1

    @testset for volume_form in (WeakForm(), FluxDifferencingForm(CentralFlux()))
        errors = zeros(FT, nlevels)
        for l = 1:nlevels
            K = 5 * 2^(l - 1)
            errors[l] = run(A, FT, N, K; volume_form, comm)
            @test errors[l] ≈ expected_error[volume_form, l] rtol = 10sqrt(eps(FT))
        end
        if nlevels > 1
            rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
            if rank == 0
                @info "Convergence rates\n" * join(
                    ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(nlevels-1)],
                    "\n",
                )
            end
            @test rates[end] ≈ N + 1 atol = 0.1
        end
    end
end
