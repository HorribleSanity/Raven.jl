using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.EulerGravity

using StaticArrays: SVector
using WriteVTK
using MPI
import CUDA
import KernelAbstractions as KA
import Adapt

struct RisingBubble <: AbstractProblem end

import Raven.BalanceLaws: boundarystate
function boundarystate(law::EulerGravityLaw, ::RisingBubble, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function risingbubble(law, x⃗, add_perturbation = true)
    FT = eltype(law)
    x, z = x⃗

    Φ = constants(law).grav * z

    cv_d = FT(719)
    cp_d = constants(law).γ * cv_d
    R_d = cp_d - cv_d

    θref = FT(300)
    p0 = FT(1e5)
    xc = FT(500)
    zc = FT(350)
    rc = FT(250)
    δθc = FT(1 / 2)

    r = sqrt((x - xc)^2 + (z - zc)^2)
    δθ = r <= rc ? δθc : zero(FT)

    θ = θref
    if add_perturbation
        θ += δθ * (1 + cos(π * r / rc)) / 2
    end
    π_exner = 1 - constants(law).grav / (cp_d * θ) * z
    ρ = p0 / (R_d * θ) * π_exner^(cv_d / R_d)

    ρu = FT(0)
    ρv = FT(0)

    T = θ * π_exner
    ρe = ρ * (cv_d * T + Φ)

    SVector(ρ, ρu, ρv, ρe)
end

function run(A, FT, N, K; volume_form = WeakForm(), outputvtk = true, comm = MPI.COMM_WORLD)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = EulerGravityLaw{FT,2}(problem = RisingBubble())

    cell = LobattoCell{FT,A}(Nq, Nq)
    vx = range(FT(0), stop = FT(1e3), length = K + 1)
    vz = range(FT(0), stop = FT(1e3), length = K + 1)

    coarsegrid = brick((vx, vz), (true, false))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)


    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RoeFlux(), comm)

    cfl = FT(1 // 3)
    dt = cfl * step(vz) / N / 330
    timeend = @isdefined(_testing) ? 10dt : FT(500)

    q = GridArray(undef, law, grid)
    q .= risingbubble.(Ref(law), points(grid))
    qref = GridArray(undef, law, grid)
    qref .= risingbubble.(Ref(law), points(grid), false)

    if outputvtk
        vtkdir = joinpath("output", "euler_gravity", "risingbubble")
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
                ρ, ρu, ρv, ρe = components(q)
                ρ_ref, ρu_ref, ρv_ref, ρe_ref = components(qref)
                vtkfile["ρ"] = Adapt.adapt(Array, (P * (ρ - ρ_ref)))
                vtkfile["ρu"] = Adapt.adapt(Array, (P * (ρu - ρu_ref)))
                vtkfile["ρv"] = Adapt.adapt(Array, (P * (ρv - ρv_ref)))
                vtkfile["ρe"] = Adapt.adapt(Array, (P * (ρe - ρe_ref)))
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
    K = 10

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

    run(A, FT, N, K)
end
