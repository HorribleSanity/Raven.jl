using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.EulerGravity

using LinearAlgebra: norm
using StaticArrays: SVector
using WriteVTK
using MPI
import CUDA
import KernelAbstractions as KA
import Adapt

struct GravityWave <: AbstractProblem end

import Raven.BalanceLaws: boundarystate
function boundarystate(law::EulerGravityLaw, ::GravityWave, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Raven.BalanceLaws.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, ::GravityWave, x⃗)
    FT = eltype(law)
    x, z = x⃗

    cv_d = FT(719)
    cp_d = constants(law).γ * cv_d
    R_d = cp_d - cv_d

    p_s = FT(1e5)
    T_ref = FT(250)

    δ = constants(law).grav / (R_d * T_ref)
    ρ_s = p_s / (T_ref * R_d)
    ρ_ref = ρ_s * exp(-δ * z)

    p_ref = ρ_ref * R_d * T_ref

    SVector(ρ_ref, p_ref)
end

function gravitywave(law, x⃗, t, add_perturbation = true)
    FT = eltype(law)
    x, z = x⃗

    Φ = constants(law).grav * z

    cv_d = FT(719)
    cp_d = constants(law).γ * cv_d
    R_d = cp_d - cv_d

    ΔT = FT(0.0001)
    H = FT(10e3)
    f = FT(0)
    L = FT(300e3)
    d = FT(5e3)
    x_c = FT(100e3)
    u_0 = FT(20)
    p_s = FT(1e5)
    T_ref = FT(250)

    g = constants(law).grav
    δ = g / (R_d * T_ref)
    c_s = sqrt(cp_d / cv_d * R_d * T_ref)
    ρ_s = p_s / (T_ref * R_d)

    if t == 0
        δT_b = ΔT * exp(-(x - x_c)^2 / d^2) * sin(π * z / H)
        δT = exp(δ * z / 2) * δT_b
        δρ_b = -ρ_s * δT_b / T_ref
        δρ = exp(-δ * z / 2) * δρ_b
        δu, δv, δw = 0, 0, 0
    else
        xp = x - u_0 * t

        δρ_b, δu_b, δv_b, δw_b, δp_b = zeros(SVector{5,Complex{FT}})
        for m in (-1, 1)
            for n = -100:100
                k_x = 2π * n / L
                k_z = π * m / H

                p_1 = c_s^2 * (k_x^2 + k_z^2 + δ^2 / 4) + f^2
                q_1 = g * k_x^2 * (c_s^2 * δ - g) + c_s^2 * f^2 * (k_z^2 + δ^2 / 4)

                α = sqrt(p_1 / 2 - sqrt(p_1^2 / 4 - q_1))
                β = sqrt(p_1 / 2 + sqrt(p_1^2 / 4 - q_1))

                fac1 = 1 / (β^2 - α^2)
                L_m1 = (-cos(α * t) / α^2 + cos(β * t) / β^2) * fac1 + 1 / (α^2 * β^2)
                L_0 = (sin(α * t) / α - sin(β * t) / β) * fac1
                L_1 = (cos(α * t) - cos(β * t)) * fac1
                L_2 = (-α * sin(α * t) + β * sin(β * t)) * fac1
                L_3 = (-α^2 * cos(α * t) + β^2 * cos(β * t)) * fac1

                if α == 0
                    L_m1 = (β^2 * t^2 - 1 + cos(β * t)) / β^4
                    L_0 = (β * t - sin(β * t)) / β^3
                end

                δρ̃_b0 =
                    -ρ_s / T_ref * ΔT / sqrt(π) * d / L *
                    exp(-d^2 * k_x^2 / 4) *
                    exp(-im * k_x * x_c) *
                    k_z *
                    H / 2im

                δρ̃_b =
                    (
                        L_3 +
                        (p_1 + g * (im * k_z - δ / 2)) * L_1 +
                        (c_s^2 * (k_z^2 + δ^2 / 4) + g * (im * k_z - δ / 2)) * f^2 * L_m1
                    ) * δρ̃_b0

                δp̃_b = -(g - c_s^2 * (im * k_z + δ / 2)) * (L_1 + f^2 * L_m1) * g * δρ̃_b0

                δũ_b = im * k_x * (g - c_s^2 * (im * k_z + δ / 2)) * L_0 * g * δρ̃_b0 / ρ_s

                δṽ_b =
                    -f * im * k_x * (g - c_s^2 * (im * k_z + δ / 2)) * L_m1 * g * δρ̃_b0 /
                    ρ_s

                δw̃_b = -(L_2 + (f^2 + c_s^2 * k_x^2) * L_0) * g * δρ̃_b0 / ρ_s

                expfac = exp(im * (k_x * xp + k_z * z))

                δρ_b += δρ̃_b * expfac
                δp_b += δp̃_b * expfac

                δu_b += δũ_b * expfac
                δv_b += δṽ_b * expfac
                δw_b += δw̃_b * expfac
            end
        end

        δρ = exp(-δ * z / 2) * real(δρ_b)
        δp = exp(-δ * z / 2) * real(δp_b)

        δu = exp(δ * z / 2) * real(δu_b)
        δv = exp(δ * z / 2) * real(δv_b)
        δw = exp(δ * z / 2) * real(δw_b)

        δT_b = T_ref * (δp_b / p_s - δρ_b / ρ_s)
        δT = exp(δ * z / 2) * real(δT_b)
    end

    ρ_ref = ρ_s * exp(-δ * z)

    ρ = ρ_ref
    T = T_ref
    u = u_0
    w = FT(0)

    if add_perturbation
        ρ += δρ
        T += δT
        u += δu
        w += δw
    end

    e_kin = (u^2 + w^2) / 2
    e_int = cv_d * T
    ρe = ρ * (e_int + e_kin + Φ)

    return SVector(ρ, ρ * u, ρ * w, ρe)
end

function run(
    A,
    FT,
    N,
    KX,
    KY;
    volume_form = WeakForm(),
    outputvtk = true,
    comm = MPI.COMM_WORLD,
)
    Nq = N + 1

    rank = MPI.Comm_rank(comm)
    law = EulerGravityLaw{FT,2}(pde_level_balance = true, problem = GravityWave())

    cell = LobattoCell{FT,A}(Nq, Nq)
    vx = range(FT(0), stop = FT(300e3), length = KX + 1)
    vz = range(FT(0), stop = FT(10e3), length = KY + 1)

    coarsegrid = brick((vx, vz), (true, false))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RoeFlux(), comm)

    cfl = FT(1 // 3)
    dt = cfl * step(vz) / N / 330
    timeend = @isdefined(_testing) ? 10dt : FT(30 * 60)

    q = GridArray(undef, law, grid)
    q .= gravitywave.(Ref(law), points(grid), FT(0))
    qref = GridArray(undef, law, grid)
    qref .= gravitywave.(Ref(law), points(grid), FT(0), false)

    if outputvtk
        vtkdir = joinpath("output", "euler_gravity", "gravitywave", "KX_$(KX)_KY_$(KY)")
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
                vtkfile["δρ"] = Adapt.adapt(Array, (P * (ρ - ρ_ref)))
                vtkfile["δρu"] = Adapt.adapt(Array, (P * (ρu - ρu_ref)))
                vtkfile["δρv"] = Adapt.adapt(Array, (P * (ρv - ρv_ref)))
                vtkfile["δρe"] = Adapt.adapt(Array, (P * (ρe - ρe_ref)))
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

    qexact = GridArray(undef, law, grid)
    qexact .= gravitywave.(Ref(law), points(grid), timeend)
    errf = map(components(q), components(qexact)) do f, fexact
        sqrt(sum(dg.MJ .* (f .- fexact) .^ 2))
    end
    norm(errf)
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

    ndof_x = 60
    ndof_y = 15
    KX_base = round(Int, ndof_x / N)
    KY_base = round(Int, ndof_y / N)

    nlevels = @isdefined(_testing) ? 1 : 2
    errors = zeros(FT, nlevels)
    for l = 1:nlevels
        KX = KX_base * 2^(l - 1)
        KY = KY_base * 2^(l - 1)
        errors[l] = run(A, FT, N, KX, KY)
    end
    if nlevels > 1
        rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
        if rank == 0
            @show errors
            @show rates
        end
    end
end
