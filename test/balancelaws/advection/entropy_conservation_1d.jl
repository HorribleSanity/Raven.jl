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

function square(law, x⃗)
    FT = eltype(law)
    ρ = @inbounds abs(x⃗[1]) < FT(1 // 2) ? FT(2) : FT(1)
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

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = CentralFlux(), comm)

    cfl = FT(1 // 4)
    dt = cfl * step(v1d) / Nq / norm(constants(law).u⃗)
    timeend = FT(5.0)

    q = GridArray(undef, law, grid)
    q .= square.(Ref(law), points(grid))
    η0 = entropyintegral(dg, q)

    normq = weightednorm(dg, q)
    if rank == 0
        @info @sprintf """Starting
        N           = %d
        K           = %d
        volume_form = %s
        norm(q)     = %.16e
        η0          = %.16e
        """ N K volume_form normq η0
    end

    odesolver = RLSRK54(dg, q, dt)

    solve!(q, timeend, odesolver)
    ηf = entropyintegral(dg, q)

    Δη = (ηf - η0) / abs(η0)

    normq = weightednorm(dg, q)
    if rank == 0
        @info @sprintf """Finished
        norm(q) = %.16e
        ηf      = %.16e
        Δη      = %.16e
        """ normq ηf Δη
    end
    Δη
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
    K = 5

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

    volume_form = FluxDifferencingForm(CentralFlux())

    if rank == 0
        @info """Using
        A           = $A
        FT          = $FT
        N           = $N
        volume_form = $volume_form
        backend     = $backend
        device      = $(KA.device(backend))
       """
    end

    Δη = run(A, FT, N, K; volume_form, comm)

    @test abs(Δη) <= 40eps(FT)
end
