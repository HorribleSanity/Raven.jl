using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.ShallowWater

using StaticArrays: SVector
using WriteVTK
using MPI
import CUDA
import KernelAbstractions as KA
import Adapt

struct BickleyJet <: AbstractProblem end

import Raven.BalanceLaws: boundarystate
function boundarystate(law::ShallowWaterLaw, ::BickleyJet, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρθ⁻ = ShallowWater.unpackstate(law, q⁻)
    ρ⁺, ρθ⁺ = ρ⁻, ρθ⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρθ⁺), aux⁻
end

function bickleyjet(law, x⃗)
    FT = eltype(law)
    x, y = x⃗

    ϵ = FT(1 / 10)
    l = FT(1 / 2)
    k = FT(1 / 2)

    U = cosh(y)^(-2)

    Ψ = exp(-(y + l / 10)^2 / (2 * (l^2))) * cos(k * x) * cos(k * y)

    u = Ψ * (k * tan(k * y) + y / (l^2))
    v = -Ψ * k * tan(k * x)

    ρ = FT(1)
    ρu = ρ * (U + ϵ * u)
    ρv = ρ * (ϵ * v)
    ρθ = ρ * sin(k * y)

    SVector(ρ, ρu, ρv, ρθ)
end

function run(A, FT, N, K; volume_form = WeakForm(), outputvtk = true, comm = MPI.COMM_WORLD)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = ShallowWaterLaw{FT,2}(problem = BickleyJet())

    cell = LobattoCell{FT,A}(Nq, Nq)
    v1d = range(FT(-2π), stop = FT(2π), length = K + 1)

    coarsegrid = brick((v1d, v1d), (true, false))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RoeFlux())

    cfl = FT(1 // 8)
    dt = cfl * step(v1d) / N / sqrt(constants(law).grav)
    timeend = @isdefined(_testing) ? 10dt : FT(200)

    q = GridArray(undef, law, grid)
    q .= bickleyjet.(Ref(law), points(grid))

    if outputvtk
        vtkdir = joinpath("output", "shallow_water", "bickleyjet")
        rank == 0 && mkpath(vtkdir)
        MPI.Barrier(comm) # Wait for directory to be created
        pvd = rank == 0 ? paraview_collection("timesteps") : nothing
    end

    do_output = function (step, time, q)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            cd(vtkdir) do
                filename = "step$(lpad(step, 6, '0'))"
                vtkfile = vtk_grid(filename, grid)
                P = toequallyspaced(cell)
                ρθ = last(components(q))
                vtkfile["ρθ"] = Adapt.adapt(Array, (P * ρθ))
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    odesolver = LSRK54(dg, q, dt)

    outputvtk && do_output(0, FT(0), q)
    solve!(q, timeend, odesolver; after_step = do_output)
    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end
end

let
    if !MPI.Initialized()
        MPI.Init(threadlevel = :multiple)
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    A = Array
    FT = Float64
    N = 3
    K = 16

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
        K       = $K
        backend = $backend
        device  = $(KA.device(backend))
       """
    end

    run(A, FT, N, K; comm)
end
