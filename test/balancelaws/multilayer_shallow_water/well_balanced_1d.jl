using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.MultilayerShallowWater

using Test
using Printf
using LinearAlgebra: norm
using StaticArrays: SVector
using MPI
import CUDA
import KernelAbstractions as KA

# this state requires 3 layers
function balancedstate(law, x)
    FT = eltype(law)

    H = SVector(FT(7) / FT(10), FT(6) / FT(10), FT(5) / FT(10))
    v = SVector(-zero(FT), -zero(FT), -zero(FT))
    b = first(balancedaux(law, x))

    h = -diff(SVector(H..., b))
    return SVector(h..., (v .* h)...)
end

function balancedaux(law, x)
    FT = eltype(law)

    x1 = first(x)

    inicenter = FT(1) / FT(2)
    r = abs(x1 - inicenter)

    b =
        r <= FT(1) / FT(10) ? FT(2) / FT(10) * (cos(10 * (x1 - FT(1) / FT(2)) * pi) + 1) :
        -zero(FT)

    return SVector(b)
end

function run(A, FT, N, K; volume_form = WeakForm(), comm = MPI.COMM_WORLD)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = MultilayerShallowWaterLaw{FT,1,3}(;
        g = FT(1),
        ρ = (FT(8) / FT(10), FT(9) / FT(10), FT(1)),
    )

    cell = LobattoCell{FT,A}(Nq)
    v1d = range(FT(0), stop = FT(1), length = K + 1)
    coarsegrid = brick((v1d,), (true,))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    auxstate = GridArray{SVector{1,FT}}(undef, grid)
    auxstate .= balancedaux.(Ref(law), points(grid))
    aux_comm_manager = commmanager(eltype(auxstate), nodecommpattern(grid); comm)
    start!(auxstate, aux_comm_manager)
    finish!(auxstate, aux_comm_manager)

    dg = DGSEM(;
        law,
        grid,
        volume_form,
        surface_numericalflux = EntropyConservativeFlux(),
        comm,
        auxstate,
    )

    cfl = FT(1 // 4)
    dt = cfl * step(v1d) / Nq^2
    timeend = FT(1 // 8)

    q = GridArray(undef, law, grid)
    q .= balancedstate.(Ref(law), points(grid))
    η0 = entropyintegral(dg, q)

    qexact = copy(q)

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

    errf = weightednorm(dg, q .- qexact)
    normq = weightednorm(dg, q)
    if rank == 0
        @info @sprintf """Finished
        norm(q)      = %.16e
        norm(q - qe) = %.16e
        ηf           = %.16e
        Δη           = %.16e
        """ normq errf ηf Δη
    end
    errf, Δη
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
    K = 50

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

    volume_form = FluxDifferencingForm(EntropyConservativeFlux())

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

    errf, Δη = run(A, FT, N, K; volume_form, comm)

    @test errf <= 40eps(FT)
    @test abs(Δη) <= 40eps(FT)
end
