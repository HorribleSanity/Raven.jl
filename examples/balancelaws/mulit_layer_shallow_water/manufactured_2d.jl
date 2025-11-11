using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.MultilayerShallowWater

using Printf
using StaticArrays: SVector
using LinearAlgebra: norm
using MPI
using CUDA
import KernelAbstractions as KA

using OrdinaryDiffEqTsit5
using OrdinaryDiffEqLowStorageRK
using Theseus

function manufacturedstate(law, x, t)
    FT = eltype(law)


    # Some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]
    ω = pi * sqrt(FT(2))

    @inbounds begin
        b = 1 + FT(1) / FT(10) * cos(ω * x[1]) + FT(1) / FT(10) * cos(ω * x[2])

        H1 = 4 + FT(1) / FT(10) * cos(ω * x[1] + t) + FT(1) / FT(10) * cos(ω * x[2] + t)
        H2 = 2 + FT(1) / FT(10) * sin(ω * x[1] + t) + FT(1) / FT(10) * sin(ω * x[2] + t)
        H3 =
            FT(15) / FT(10) +
            FT(1) / FT(10) * cos(ω * x[1] + t) +
            FT(1) / FT(10) * cos(ω * x[2] + t)

        v1 = SVector(FT(8 // 10), FT(1))
        v2 = SVector(FT(8 // 10), FT(1))
        v3 = SVector(FT(8 // 10), FT(1))

        h1 = H1 - H2
        h2 = H2 - H3
        h3 = H3 - b

        vh1 = v1 * h1
        vh2 = v2 * h2
        vh3 = v3 * h3
    end

    return SVector(h1, h2, h3, vh1..., vh2..., vh3...)
end


function manufacturedaux(law, x)
    FT = eltype(law)

    ω = pi * sqrt(FT(2))
    @inbounds begin
        b = 1 + FT(1) / FT(10) * cos(ω * x[1]) + FT(1) / FT(10) * cos(ω * x[2])
    end

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

        @inbounds begin
            x = SVector(aux[2], aux[3])

            # h1
            du1 = (
                -(FT(1) / FT(10)) * cos(t + x[2] * ω) -
                (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                (FT(1) / FT(10)) * cos(t + x[1] * ω) -
                (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                (FT(8) / FT(10)) * (
                    -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) - (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
            )

            # h2
            du2 = (
                (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                (FT(1) / FT(10)) * cos(t + x[1] * ω) +
                (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) +
                (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
            )

            # h3
            du3 = (
                -(FT(1) / FT(10)) * sin(t + x[1] * ω) -
                (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                (FT(1) / FT(10)) * sin(x[2] * ω) * ω +
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * sin(x[1] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω
                ) - (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
            )

            # h1_v1
            du4 = (
                (FT(8) / FT(10)) * (
                    -(FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) +
                (FT(8) / FT(10)) * (
                    -(FT(1) / FT(10)) * cos(t + x[2] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                ) +
                (FT(8) / FT(10))^2 * (
                    -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) +
                g *
                (
                    2 + (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) +
                (FT(1) / FT(10)) *
                g *
                (
                    2 + (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                cos(t + x[1] * ω) *
                ω
            )

            # h1_v2
            du5 = (
                -(FT(1) / FT(10)) * cos(t + x[2] * ω) -
                (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                (FT(1) / FT(10)) * cos(t + x[1] * ω) -
                (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                (FT(8) / FT(10)) * (
                    -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) - (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω +
                (FT(1) / FT(10)) *
                g *
                (
                    2 + (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                cos(t + x[2] * ω) *
                ω +
                g *
                (
                    2 + (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    -(FT(1) / FT(10)) * cos(t + x[2] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                )
            )

            # h2_v1
            du6 = (
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) +
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                ) +
                (FT(8) / FT(10))^2 * (
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω +
                    (FT(9) / FT(10)) * (
                        -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω -
                        (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                    )
                ) +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                )
            )

            # h2_v2
            du7 = (
                (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                (FT(1) / FT(10)) * cos(t + x[1] * ω) +
                (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                ) +
                (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                ) +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(t + x[2] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) +
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    (FT(9) / FT(10)) * (
                        -(FT(1) / FT(10)) * cos(t + x[2] * ω) * ω -
                        (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                    ) - (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                )
            )

            # h3_v1
            du8 = (
                (FT(8) / FT(10)) * (
                    -(FT(1) / FT(10)) * sin(t + x[1] * ω) -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω)
                ) +
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * sin(x[2] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                ) +
                (FT(8) / FT(10))^2 * (
                    (FT(1) / FT(10)) * sin(x[1] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω
                ) +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(x[1] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    (FT(1) / FT(10)) * sin(x[1] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω
                ) +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(x[1] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    -(FT(1) / FT(10)) * sin(x[1] * ω) * ω +
                    (FT(9) / FT(11)) * (
                        -(FT(1) / FT(10)) * sin(t + x[1] * ω) * ω -
                        (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                    ) +
                    (FT(10) / FT(11)) * (
                        (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω +
                        (FT(1) / FT(10)) * cos(t + x[1] * ω) * ω
                    )
                )
            )

            # h3_v2
            du9 = (
                -(FT(1) / FT(10)) * sin(t + x[1] * ω) -
                (FT(1) / FT(10)) * sin(t + x[2] * ω) +
                (FT(1) / FT(10)) * sin(x[2] * ω) * ω +
                (FT(8) / FT(10)) * (
                    (FT(1) / FT(10)) * sin(x[1] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[1] * ω) * ω
                ) - (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(x[1] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    (FT(9) / FT(11)) * (
                        -(FT(1) / FT(10)) * cos(t + x[2] * ω) * ω -
                        (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                    ) +
                    (FT(10) / FT(11)) * (
                        (FT(1) / FT(10)) * cos(t + x[2] * ω) * ω +
                        (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                    ) - (FT(1) / FT(10)) * sin(x[2] * ω) * ω
                ) +
                g *
                (
                    (FT(1) / FT(2)) - (FT(1) / FT(10)) * cos(x[1] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[2] * ω) -
                    (FT(1) / FT(10)) * cos(x[2] * ω) +
                    (FT(1) / FT(10)) * cos(t + x[1] * ω)
                ) *
                (
                    (FT(1) / FT(10)) * sin(x[2] * ω) * ω -
                    (FT(1) / FT(10)) * sin(t + x[2] * ω) * ω
                )
            )


        end

        dq .+= SVector(du1, du2, du3, du4, du5, du6, du7, du8, du9)
    end
end

function build(
    A,
    FT,
    N,
    K;
    volume_form = FluxDifferencingForm(EntropyConservativeFlux()),
    comm = MPI.COMM_WORLD,
)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = MultilayerShallowWaterLaw{FT,2,3}(
        g = FT(11 // 10),
        ρ = SVector(FT(9 // 10), FT(1), FT(11 // 10)),
        problem = Manufactured(),
    )

    cell = LobattoCell{FT,A}(Nq, Nq)
    v1d = range(FT(0), stop = sqrt(FT(2)), length = K + 1)
    coarsegrid = brick((v1d, v1d), (true, true))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    auxstate = GridArray{SVector{1 + ndims(cell),FT}}(undef, grid)
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

    cfl = FT(1 // 8)
    dt = cfl * 2step(v1d) / Nq^2
    timeend = FT(1)

    numberofsteps = cld(timeend, dt)
    dt = FT(timeend / numberofsteps)

    tspan = (0, timeend)
    ode = Raven.semidiscretize(dg, tspan)
    parent(ode.u0) .= manufacturedstate.(Ref(law), points(grid), FT(0))

    qexact = GridArray(undef, law, grid)
    qexact .= manufacturedstate.(Ref(law), points(grid), timeend)

    return ode, dt, qexact
end


begin
    if !MPI.Initialized()
        MPI.Init(threadlevel = :multiple)
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    A = Array
    FT = Float64
    N = 4
    nlevels = 1
    volume_form = FluxDifferencingForm(EntropyConservativeFlux())

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

    errors = zeros(FT, nlevels)
    for l = 1:nlevels
        K = 2 * 2^(l)

        global ode, dt, qexact = build(A, FT, N, K; volume_form, comm)

        dg = ode.p

        normq = weightednorm(dg, parent(ode.u0))

        if rank == 0
            @info @sprintf """Starting
            N                   = %d
            K                   = %d
            FT                  = %s
            A                   = %s
            backend             = %s
            integration_testing = %s
            norm(q)             = %.16e
        """ N K FT A KA.get_backend(parent(ode.u0)) volume_form normq
        end


        global sol_lsrk = solve(
            ode, CarpenterKennedy2N54();
            dt,
            save_everystep=true,
            # callback = callbacks,
            adaptive=false
        )

        q = parent(last(sol_lsrk.u))
        errf = weightednorm(dg, q .- qexact)
        normq = weightednorm(dg, q)

        if rank == 0
            @info @sprintf """CarpenterKennedy2N54
            norm(q)      = %.16e
            norm(q - qe) = %.16e
            """ normq errf
        end

        global sol_tsit5 = solve(
            ode, Tsit5();
            dt,
            save_everystep=true,
            # callback = callbacks,
            adaptive=false
        )

        q = parent(last(sol_tsit5.u))
        errf = weightednorm(dg, q .- qexact)
        normq = weightednorm(dg, q)

        if rank == 0
            @info @sprintf """Tsit5
            norm(q)      = %.16e
            norm(q - qe) = %.16e
            """ normq errf
        end

        # TODO: CFL and other callbacks
        global sol = solve(
            ode, Theseus.ROS2();
            dt=100dt,
            #verbose=1,
            krylov_algo=:gmres,
            assume_p_const=false,
        )

        q = parent(last(sol.u))
        errf = weightednorm(dg, q .- qexact)
        normq = weightednorm(dg, q)

        if rank == 0
            @info @sprintf """Theseus
            norm(q)      = %.16e
            norm(q - qe) = %.16e
            """ normq errf
        end



        q = parent(ode.u0)
        odesolver = LSRK54(dg, q, dt)
        timeend = last(ode.tspan)
        Raven.BalanceLaws.solve!(q, timeend, odesolver)

        errf = weightednorm(dg, q .- qexact)
        normq = weightednorm(dg, q)

        if rank == 0
            @info @sprintf """Raven LSRK54
            norm(q)      = %.16e
            norm(q - qe) = %.16e
            """ normq errf
        end

        errors[l] = errf
    end

    @show errors

    if nlevels > 1
        rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
        if rank == 0
            @info "Convergence rates\n" * join(
                ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(nlevels-1)],
                "\n",
            )
        end
    end


end
