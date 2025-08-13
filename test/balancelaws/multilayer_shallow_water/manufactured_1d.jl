using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.MultilayerShallowWater

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

function manufacturedstate(law, x, t)
    FT = eltype(law)

    x1 = first(x)

    # Some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]
    ω = pi * sqrt(FT(2))
    b = 1 + 1 // 10 * cos(ω * x1)

    H1 = 4 + FT(1) / FT(10) * cos(ω * x1 + t)
    H2 = 2 + FT(1) / FT(10) * sin(ω * x1 + t)
    H3 = FT(15) / FT(10) + FT(1) / FT(10) * cos(ω * x1 + t)
    v1 = FT(9 // 10)
    v2 = FT(1)
    v3 = FT(11 // 10)

    h1 = H1 - H2
    h2 = H2 - H3
    h3 = H3 - b

    return SVector(h1, h2, h3, v1 * h1, v2 * h2, v3 * h3)
end


function manufacturedaux(law, x)
    FT = eltype(law)

    x1 = first(x)
    ω = pi * sqrt(FT(2))
    b = 1 + 1 // 10 * cos(ω * x1)

    return SVector(b, x...)
end


struct Manufactured <: AbstractProblem end

function BalanceLaws.source!(
    law::MultilayerShallowWaterLaw,
    ::Manufactured,
    dq,
    q,
    aux,
    dim,
    directions,
    t,
)
    if dim ∈ directions
        FT = eltype(law)

        # Some derivative simplify because this manufactured solution
        # velocity is taken to be constant
        ω = pi * sqrt(FT(2))
        g = constants(law).g

        x1 = aux[2]

        du1 =
            -FT(1 // 10) * sin(t + x1 * ω) - FT(1 // 10) * cos(t + x1 * ω) +
            FT(9 // 10) *
            (-FT(1 // 10) * sin(t + x1 * ω) * ω - FT(1 // 10) * cos(t + x1 * ω) * ω)

        du2 =
            FT(1 // 10) * sin(t + x1 * ω) +
            FT(1 // 10) * cos(t + x1 * ω) +
            FT(1 // 10) * sin(t + x1 * ω) * ω +
            FT(1 // 10) * cos(t + x1 * ω) * ω

        du3 =
            -FT(1 // 10) * sin(t + x1 * ω) +
            FT(11 // 10) *
            (FT(1 // 10) * sin(x1 * ω) * ω - FT(1 // 10) * sin(t + x1 * ω) * ω)

        du4 =
            FT(9 // 10) * (-FT(1 // 10) * sin(t + x1 * ω) - FT(1 // 10) * cos(t + x1 * ω)) +
            FT(81 // 100) *
            (-FT(1 // 10) * sin(t + x1 * ω) * ω - FT(1 // 10) * cos(t + x1 * ω) * ω) +
            g *
            (2 - FT(1 // 10) * sin(t + x1 * ω) + FT(1 // 10) * cos(t + x1 * ω)) *
            (-FT(1 // 10) * sin(t + x1 * ω) * ω - FT(1 // 10) * cos(t + x1 * ω) * ω) +
            FT(1 // 10) *
            g *
            (2 - FT(1 // 10) * sin(t + x1 * ω) + FT(1 // 10) * cos(t + x1 * ω)) *
            cos(t + x1 * ω) *
            ω

        du5 =
            FT(1 // 10) * sin(t + x1 * ω) +
            FT(1 // 10) * cos(t + x1 * ω) +
            FT(1 // 10) * sin(t + x1 * ω) * ω +
            FT(1 // 10) * cos(t + x1 * ω) * ω +
            g *
            (FT(1 // 2) + FT(1 // 10) * sin(t + x1 * ω) - FT(1 // 10) * cos(t + x1 * ω)) *
            (FT(1 // 10) * sin(t + x1 * ω) * ω + FT(1 // 10) * cos(t + x1 * ω) * ω) +
            g *
            (FT(1 // 2) + FT(1 // 10) * sin(t + x1 * ω) - FT(1 // 10) * cos(t + x1 * ω)) *
            (
                -FT(1 // 10) * sin(t + x1 * ω) * ω +
                FT(9 // 10) *
                (-FT(1 // 10) * sin(t + x1 * ω) * ω - FT(1 // 10) * cos(t + x1 * ω) * ω)
            )

        du6 =
            -FT(11 // 100) * sin(t + x1 * ω) +
            FT(121 / 100) *
            (FT(1 // 10) * sin(x1 * ω) * ω - FT(1 // 10) * sin(t + x1 * ω) * ω) +
            g *
            (FT(1 // 2) - FT(1 // 10) * cos(x1 * ω) + FT(1 // 10) * cos(t + x1 * ω)) *
            (
                -FT(1 // 10) * sin(x1 * ω) * ω +
                FT(10 // 11) *
                (FT(1 // 10) * sin(t + x1 * ω) * ω + FT(1 // 10) * cos(t + x1 * ω) * ω) +
                FT(9 // 11) *
                (-FT(1 // 10) * sin(t + x1 * ω) * ω - FT(1 // 10) * cos(t + x1 * ω) * ω)
            ) +
            g *
            (FT(1 // 2) - FT(1 // 10) * cos(x1 * ω) + FT(1 // 10) * cos(t + x1 * ω)) *
            (FT(1 // 10) * sin(x1 * ω) * ω - FT(1 // 10) * sin(t + x1 * ω) * ω)

        dq .+= SVector(du1, du2, du3, du4, du5, du6)
    end
end

function run(
    A,
    FT,
    N,
    K;
    volume_form = FluxDifferencingForm(EntropyConservativeFlux()),
    comm = MPI.COMM_WORLD,
)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = MultilayerShallowWaterLaw{FT,1,3}(
        g = FT(11 // 10),
        ρ = SVector(FT(9 // 10), FT(1), FT(11 // 10)),
        problem = Manufactured(),
    )

    cell = LobattoCell{FT,A}(Nq)
    v1d = range(FT(0), stop = sqrt(FT(2)), length = K + 1)
    coarsegrid = brick((v1d,), (true,))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    auxstate = GridArray{SVector{2,FT}}(undef, grid)
    auxstate .= manufacturedaux.(Ref(law), points(grid))
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

    cfl = FT(1 // 2)
    dt = cfl * 2step(v1d) / Nq^2
    timeend = FT(1)

    numberofsteps = cld(timeend, dt)
    dt = FT(timeend / numberofsteps)

    q = GridArray(undef, law, grid)
    q .= manufacturedstate.(Ref(law), points(grid), FT(0))

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
    qexact .= manufacturedstate.(Ref(law), points(grid), timeend)
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
    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 1] =
        0.030582654324792385
    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 2] =
        0.00010693547675592389
    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 3] =
        3.803063008876906e-6
    expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 4] =
        8.931055218967449e-8

    nlevels = integration_testing ? 4 : 1

    @testset for volume_form in (FluxDifferencingForm(EntropyConservativeFlux()),)
        errors = zeros(FT, nlevels)
        for l = 1:nlevels
            K = 2 * 2^(l - 1)
            errf = run(A, FT, N, K; volume_form, comm)
            errors[l] = errf
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
            @test rates[end] ≈ N + 1 atol = 0.5
        end
    end
end
