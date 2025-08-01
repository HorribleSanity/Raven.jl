using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.Euler

using StaticArrays: SVector
using WriteVTK
using MPI
import CUDA
import KernelAbstractions as KA
import Adapt

struct Sod <: AbstractProblem end

import Raven.BalanceLaws: boundarystate
function boundarystate(law::EulerLaw, ::Sod, n⃗, q⁻, aux⁻, bctag)
    FT = eltype(law)
    bctag == 1 ? sod(law, SVector(FT(0)), SVector(FT(0))) :
    sod(law, SVector(FT(1)), SVector(FT(1))),
    aux⁻
end

function sod(law, _, xave)
    FT = eltype(law)
    ρ = xave[1] < 1 // 2 ? 1 : 1 // 8
    ρu⃗ = SVector(FT(0))
    p = xave[1] < 1 // 2 ? 1 : 1 // 10
    ρe = Euler.energy(law, ρ, ρu⃗, p)
    SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, K; volume_form = WeakForm(), outputvtk = true, comm = MPI.COMM_WORLD)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = EulerLaw{FT,1}(problem = Sod())

    cell = LobattoCell{FT,A}(Nq)
    v1d = range(FT(0), stop = FT(1), length = K + 1)

    coarsegrid = brick((v1d,), (false,))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RusanovFlux(), comm)

    cfl = FT(1 // 4)
    dt = cfl * step(v1d) / N / Euler.soundspeed(law, FT(1), FT(1))
    timeend = FT(2 // 10)

    xave = sum(A(points(grid)), dims = 1) ./ Nq

    q = GridArray(undef, law, grid)
    q .= sod.(Ref(law), points(grid), xave)

    if outputvtk
        vtkdir = joinpath("output", "euler", "sod")
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
                ρ, ρu, ρe = components(q)
                p = Euler.pressure.(Ref(law), ρ, ρu, ρe)
                u = ρu ./ ρ

                ρ = last(components(q))
                vtkfile["ρ"] = Adapt.adapt(Array, (P * ρ))
                vtkfile["ρu"] = Adapt.adapt(Array, (P * ρu))
                vtkfile["ρe"] = Adapt.adapt(Array, (P * ρe))
                vtkfile["u"] = Adapt.adapt(Array, (P * u))
                vtkfile["p"] = Adapt.adapt(Array, (P * p))
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
    N = 4
    K = 32
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

    if rank == 0
        @info """Using
        A           = $A
        FT          = $FT
        N           = $N
        K           = $K
        volume_form = $volume_form
        backend     = $backend
        device      = $(KA.device(backend))
       """
    end

    errf = run(A, FT, N, K; volume_form)
end
