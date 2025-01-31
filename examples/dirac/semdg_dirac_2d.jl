#--------------------------------Markdown Language Header-----------------------
# # Dirac Equation
#
# ```
## @MISC{Bal2018,
#    author = {Guillaume Bal},
#    title = {Topological Protection Perturbed of Edge States},
#    year = {2018},
#    url = {arXiv:1709.00605v2},
# }
# ```
#
#jl #FIXME: add convergence study
#jl #FIXME: optimize kernels
#jl #FIXME: add introduction to dirac equation latex
#--------------------------------Markdown Language Header-----------------------
using Base: sign_mask
using Adapt
using MPI
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Printf
using Raven
using StaticArrays
using WriteVTK

using SpecialFunctions

import Base.abs2
function abs2(v::SVector{2,ComplexF64})
    return abs2.(v)
end

function initialCondition(x::SVector{2})
    FT = eltype(x)
    CT = Complex{FT}
    return exp(-(x[1] / 0.3)^2) * exp(-(x[2] / 0.3)^2) .* SVector{2,CT}(1, 1)
end

# ∂ₜψ = -σ₁∂ₓψ-σ₂∂ᵥψ-im(y)σ₃ψ
@kernel function rhs_volume_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    materialparams,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering 

    # C compile time data corrisponding to cells per workgroup
    lDT1 = @localmem eltype(eltype(q)) (N[1], N[1])
    lDT2 = @localmem eltype(eltype(q)) (N[2], N[2])
    lU = @localmem eltype(q) (N..., C) # local solution at prod(N) dofs in C cells of domain

    # Pauli Matrices: SA denotes StaticArrays
    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1) # Sets stride in j
            if j + sj <= N[1] && c1 == 0x1 # localmemory is shared amongst a workgroup so only c1 needs to work
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && c1 == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        # data (i, j) in global index cell c
        qijc = q[i, j, c]
        lU[i, j, c1] = qijc
    end

    @synchronize

    # Here a thread will compute the (i,j)th component of the dq matrix where i,j is the dof and c is the global cell index
    @inbounds begin
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]
        wJdRdXijc = wJijc * dRdXijc
        matparamijc = materialparams[i, j, c]


        @unroll for m = 0x1:N[1]
            dqijc_update -= wJdRdXijc[1] * lDT1[i, m] * σ₁ * lU[m, j, c1] # -rₓDᵣσ₁u
            dqijc_update -= wJdRdXijc[3] * lDT1[i, m] * σ₂ * lU[m, j, c1] # -rᵥDᵣσ₂u
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[2] * lDT2[j, n] * σ₁ * lU[i, n, c1] # -sₓDₛσ₁u 
            dqijc_update -= wJdRdXijc[4] * lDT2[j, n] * σ₂ * lU[i, n, c1] # -sᵥDₛσ₂u
        end
        dq[i, j, c] += invwJijc * dqijc_update - im * matparamijc * σ₃ * lU[i, j, c1]
    end
end

@kernel function rhs_surface_kernel!(
    dq,
    q,
    vmapM,
    vmapP,
    bc,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
) where {N,C}
    ij, c1 = @index(Local, NTuple)
    _, c = @index(Global, NTuple)
    lqflux = @localmem eltype(dq) (N..., C)

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    @inbounds if ij <= N[1]
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, c1] = zero(eltype(lqflux))
        end
    end

    @synchronize

    @inbounds if ij <= N[2]
        # Faces with r=-1 : West
        i = 1
        j = ij
        fid = j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf / 2

        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * (qM - qP) + nf[2] * σ₂ * (qM - qP))

        # Faces with r=1 : East 
        i = N[1]
        j = ij
        fid = N[2] + j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf / 2

        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * (qM - qP) + nf[2] * σ₂ * (qM - qP))
    end

    @synchronize

    @inbounds if ij <= N[1]
        # Faces with s=-1 : South 
        i = ij
        j = 1
        fid = 2N[2] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = bc[3, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 2
            qP = (qM - σ₂ * qM) / 2
        end

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf / 2
        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * (qM - qP) + nf[2] * σ₂ * (qM - qP))

        # Faces with s=1 : North
        i = ij
        j = N[2]
        fid = 2N[2] + N[1] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = bc[4, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 2
            qP = (qM + σ₂ * qM) / 2
        end

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * (qM - qP) + nf[2] * σ₂ * (qM - qP))
    end

    @synchronize

    # RHS update
    i = ij
    @inbounds if i <= N[1]
        @unroll for j = 1:N[2]
            dq[i, j, c] += lqflux[i, j, c1]
        end
    end
end

function coefmap!(materialparams, gridpoints; m = 1.0)
    iMax, jMax, eMax = size(gridpoints)
    for e = 1:eMax
        for i = 1:iMax, j = 1:jMax
            materialparams[i, j, e] = last(gridpoints[i, j, e]) >= 0.0 ? m : -m
        end
    end
end

function rhs!(dq, q, grid, invwJ, DT, materialparams, bc, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)
    @infiltrate
    start!(q, cm)

    C = max(512 ÷ prod(size(cell)), 1)
    rhs_volume_kernel!(backend, (size(cell)..., C))(
        dq,
        q,
        dRdX,
        wJ,
        invwJ,
        DT,
        materialparams,
        Val(size(cell)),
        Val(C);
        ndrange = size(dq),
    )


    finish!(q, cm)

    J = maximum(size(cell))
    C = max(128 ÷ J, 1)
    # J  x C workgroup sizes to evaluate  multiple elements on one wg.
    rhs_surface_kernel!(backend, (J, C))(
        dq,
        viewwithghosts(q),
        fm.vmapM,
        fm.vmapP,
        bc,
        n,
        wsJ,
        invwJ,
        Val(size(cell)),
        Val(C);
        ndrange = (J, last(size(dq))),
    )
end

function run(
    ic,
    FT,
    AT,
    N,
    K,
    L;
    outputvtk = false,
    vtkdir = "output",
    comm = MPI.COMM_WORLD,
)
    rank = MPI.Comm_rank(comm)
    cell = LobattoCell{FT,AT}((N .+ 1)...)
    coordinates = ntuple(_ -> range(FT(-1), stop = FT(1), length = K + 1), 2)
    periodicity = (true, false)
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(gm)

    timeend = FT(2π)

    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates))
    dt = cfl * dx / (maximum(N))^2

    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps

    RKA = (
        FT(0),
        FT(-567301805773 // 1357537059087),
        FT(-2404267990393 // 2016746695238),
        FT(-3550918686646 // 2091501179385),
        FT(-1275806237668 // 842570457699),
    )
    RKB = (
        FT(1432997174477 // 9575080441755),
        FT(5161836677717 // 13612068292357),
        FT(1720146321549 // 2090206949498),
        FT(3134564353537 // 4481467310338),
        FT(2277821191437 // 14882151754819),
    )

    if outputvtk
        rank == 0 && mkpath(vtkdir)
        pvd = rank == 0 ? paraview_collection("timesteps") : nothing
    end

    do_output = function (step, time, q)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            cd(vtkdir) do
                filename = "step$(lpad(step, 6, '0'))"
                vtkfile = vtk_grid(filename, grid)
                P = toequallyspaced(cell)
                vtkfile["|q|²"] = Adapt.adapt(Array, P * abs2.(q))
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    #jl # initialize state
    q = ic.(points(grid))

    #jl # storage for RHS
    dq = similar(q)
    dq .= Ref(zero(eltype(q)))

    #jl # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)
    #jl # precompute derivative transpose
    DT = derivatives_1d(cell)

    materialparams_H = zeros(size(wJ))
    gridpoints_H = adapt(Array, grid.points)
    coefmap!(materialparams_H, gridpoints_H; m = 1.0)
    materialparams = adapt(AT, materialparams_H)

    # adjust boundary code
    bc_H = adapt(Array, boundarycodes(grid))
    for j = 1:size(bc_H, 2)
        if last(gridpoints_H[end, end, j]) == 1  #North
            bc_H[4, j] = 2
        end

        if last(gridpoints_H[1, 1, j]) == -1 # South
            bc_H[3, j] = 2
        end
    end

    bc = adapt(AT, bc_H)

    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #jl # initial output
    step = 0
    time = FT(0)

    do_output(step, time, q)

    for step = 1:numberofsteps
        if time + dt > timeend
            dt = timeend - time
        end

        for stage in eachindex(RKA, RKB)
            @. dq *= RKA[stage]
            rhs!(dq, q, grid, invwJ, DT, materialparams, bc, cm)
            @. q += RKB[stage] * dt * dq
        end
        time += dt

        do_output(step, time, q)
    end

    #jl # final output
    do_output(numberofsteps, timeend, q)
    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end
end

let
    FT = Float64
    AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array
    N = (2, 2)

    cell = LobattoCell{FT,AT}((N .+ 1)...)
    gm = GridManager(cell, Raven.brick(3, 3); min_level = 1)
    grid = generate(gm)

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
        CUDA.allowscalar(false)
    end

    if rank == 0
        @info """Configuration:
            precision        = $FT
            polynomial order = $N
            array type       = $AT
        """
    end

    #jl # visualize solution of dirac wave 
    K = 2
    L = 0
    vtkdir = "vtk_semidg_dirac_2d$(K)x$(K)_L$(L)"
    if rank == 0
        @info """Starting Gaussian advection with:
            ($K, $K) coarse grid
            $L refinement level
        """
    end

    run(initialCondition, FT, AT, N, K, L; outputvtk = true, vtkdir, comm)
    rank == 0 && @info "Finished, vtk output written to $vtkdir"
end
