#--------------------------------Markdown Language Header-----------------------
# # Advection equation
# The Volume and Surface kernels are adapted from libParanumal.
# ```
# @MISC{ChalmersKarakusAustinSwirydowiczWarburton2020,
#    author = {Chalmers, N. and Karakus, A. and Austin, A. P.
#       and Swirydowicz, K. and Warburton, T.},
#    title = {{libParanumal}: a performance portable high-order
#       finite element library},
#    year = {2022},
#    url = {https://github.com/paranumal/libparanumal},
#    doi = {10.5281/zenodo.4004744},
#    note = {Release 0.5.0}
# }
# ```
#--------------------------------Markdown Language Header-----------------------
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

#jl advection velocity
const v = SVector(1, 1)

#jl Gaussian initial condition
function gaussian(x, t)
    FT = eltype(x)
    xp = mod1.(x .- v .* t, FT(2π))
    xc = SVector{2,FT}(π, π)
    exp(-2norm(xp .- xc)^2)
end

#jl # sine initial condition
sineproduct(x, t) = prod(sin.(x .- v .* t))

meshwarp(x) =
    SVector(x[1] + sin(x[1] / 2) * sin(x[2]), x[2] - sin(x[1]) * sin(x[2] / 2) / 2)

@kernel function rhs_volume_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    v,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple)
    _, _, c = @index(Global, NTuple)

    lDT1 = @localmem eltype(q) (N[1], N[1])
    lDT2 = @localmem eltype(q) (N[2], N[2])
    lF = @localmem eltype(q) (N..., C)
    lG = @localmem eltype(q) (N..., C)

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1)
            if j + sj <= N[1] && c1 == 0x1
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && c1 == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        qijc = q[i, j, c]
        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]

        fluxijc = (qijc * v)
        vdRdXijc = wJijc * (dRdXijc * fluxijc)

        lF[i, j, c1] = vdRdXijc[1]
        lG[i, j, c1] = vdRdXijc[2]
    end

    @synchronize

    @inbounds begin
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        @unroll for m = 0x1:N[1]
            dqijc_update += lDT1[i, m] * lF[m, j, c1]
        end

        @unroll for n = 0x1:N[2]
            dqijc_update += lDT2[j, n] * lG[i, n, c1]
        end

        dq[i, j, c] += invwJijc * dqijc_update
    end
end

@kernel function rhs_surface_kernel!(
    dq,
    q,
    vmapM,
    vmapP,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
) where {N,C}

    ij, c1 = @index(Local, NTuple)
    _, c = @index(Global, NTuple)

    lqflux = @localmem eltype(dq) (N..., C)

    @inbounds if ij <= N[1]
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, c1] = -zero(eltype(lqflux))
        end
    end

    @synchronize

    @inbounds if ij <= N[2]
        #jl # Faces with r=-1
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
        lqflux[i, j, c1] +=
            fscale * ((nf ⋅ (v * qM)) + (nf ⋅ (v * qP)) - abs(nf ⋅ v) * (qP - qM))

        #jl # Faces with r=1
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
        lqflux[i, j, c1] +=
            fscale * ((nf ⋅ (v * qM)) + (nf ⋅ (v * qP)) - abs(nf ⋅ v) * (qP - qM))
    end

    @synchronize

    @inbounds if ij <= N[1]
        #jl # Faces with s=-1
        i = ij
        j = 1
        fid = 2N[2] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf / 2
        lqflux[i, j, c1] +=
            fscale * ((nf ⋅ (v * qM)) + (nf ⋅ (v * qP)) - abs(nf ⋅ v) * (qP - qM))

        #jl # Faces with s=1
        i = ij
        j = N[2]
        fid = 2N[2] + N[1] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf / 2
        lqflux[i, j, c1] +=
            fscale * ((nf ⋅ (v * qM)) + (nf ⋅ (v * qP)) - abs(nf ⋅ v) * (qP - qM))
    end

    @synchronize

    #jl # RHS update
    i = ij
    @inbounds if i <= N[1]
        @unroll for j = 1:N[2]
            dq[i, j, c] -= lqflux[i, j, c1]
        end
    end
end

function rhs!(dq, q, grid, invwJ, DT, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)

    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    start!(q, cm)

    C = max(512 ÷ prod(size(cell)), 1)
    rhs_volume_kernel!(backend, (size(cell)..., C))(
        dq,
        q,
        dRdX,
        wJ,
        invwJ,
        DT,
        v,
        Val(size(cell)),
        Val(C);
        ndrange = size(dq),
    )

    finish!(q, cm)

    J = maximum(size(cell))
    C = max(512 ÷ J, 1)
    rhs_surface_kernel!(backend, (J, C))(
        dq,
        viewwithghosts(q),
        fm.vmapM,
        fm.vmapP,
        n,
        wsJ,
        invwJ,
        Val(size(cell)),
        Val(C);
        ndrange = (J, last(size(dq))),
    )

end

function run(
    solution,
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
    coordinates = ntuple(_ -> range(FT(0), stop = FT(2π), length = K + 1), 2)
    periodicity = (true, true)
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(meshwarp, gm)

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
                vtkfile["q"] = Adapt.adapt(Array, P * q)
                vtkfile["CellNumber"] = (1:length(grid)) .+ offset(grid)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    #jl # initialize state
    q = solution.(points(grid), FT(0))

    #jl # storage for RHS
    dq = similar(q)
    dq .= 0

    # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)
    #jl # precompute derivative transpose
    DT = transpose.(derivatives_1d(cell))

    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #jl # initial output
    step = 0
    time = FT(0)
    do_output(step, time, q)

    #jl # time integration
    for step = 1:numberofsteps
        if time + dt > timeend
            dt = timeend - time
        end
        for stage in eachindex(RKA, RKB)
            @. dq *= RKA[stage]
            rhs!(dq, q, grid, invwJ, DT, cm)
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

    # compute error
    _, _, wJ = components(first(volumemetrics(grid)))
    qexact = solution.(points(grid), timeend)
    #jl # TODO add sum to GridArray so the following reduction is on the device
    errf = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* (q .- qexact) .^ 2)), +, comm))

    return errf
end

let
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
        CUDA.allowscalar(false)
    end

    FT = Float64
    N = (4, 5)

    #jl # run on the GPU if possible
    AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array

    if rank == 0
        @info """Configuration:
            precision        = $FT
            polynomial order = $N
            array type       = $AT
        """
    end

    #jl # visualize solution of advected Gaussian
    K = 2
    L = 2
    vtkdir = "vtk_semdg_advection_2d$(K)x$(K)_L$(L)"
    if rank == 0
        @info """Starting Gaussian advection with:
            ($K, $K) coarse grid
            $L refinement level
        """
    end

    run(gaussian, FT, AT, N, K, L; outputvtk = true, vtkdir, comm)
    rank == 0 && @info "Finished, vtk output written to $vtkdir"

    #jl # run convergence study using a simple sine field
    rank == 0 && @info "Starting convergence study"
    numlevels = @isdefined(_testing) ? 2 : 5
    err = zeros(FT, numlevels)
    for l = 1:numlevels
        L = l - 1
        K = 4
        totalcells = (K * 2^L, K * 2^L)
        err[l] = run(sineproduct, FT, AT, N, K, L; comm)
        if rank == 0
            @info @sprintf(
                "Level %d, cells = (%2d, %2d), error = %.16e",
                l,
                totalcells...,
                err[l]
            )
        end
    end
    rates = log2.(err[1:(numlevels-1)] ./ err[2:numlevels])
    if rank == 0
        @info "Convergence rates:\n" * join(
            ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(numlevels-1)],
            "\n",
        )
    end

end
