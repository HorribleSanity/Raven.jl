#--------------------------------Markdown Language Header-----------------------
# # 2D Acoustic Equation
#FIXME: add convergence study
#FIXME: add to docs
#FIXME: optimize kernels
#--------------------------------Markdown Language Header-----------------------
using Base: sign_mask
using Adapt
using MPI
using CUDA
using LinearAlgebra
using Printf
using Raven
using StaticArrays
using WriteVTK

const outputvtk = false

function initialcondition(x::SVector{2})
    FT = eltype(x)
    p = exp(-(x[1]^2 + x[2]^2) / 0.08)
    uˣ = zero(FT)
    uʸ = zero(FT)
    return SVector{3,FT}(p, uˣ, uʸ)
end

function rhs_volume_kernel!(dq, q, dRdX, wJ, invwJ, DT, ::Val{N}, ::Val{C}) where {N,C}
    i, j, cl = threadIdx()
    c = (blockIdx().x - 1) * blockDim().z + cl

    lDT1 = CuStaticSharedArray(Float64, (N[1], N[1]))
    lDT2 = CuStaticSharedArray(Float64, (N[2], N[2]))

    lp = CuStaticSharedArray(Float64, (N..., C))
    lux = CuStaticSharedArray(Float64, (N..., C))
    luy = CuStaticSharedArray(Float64, (N..., C))

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1)
            if j + sj <= N[1] && cl == 0x1
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && cl == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        lp[i, j, cl] = q[i, j, 0x1, c]
        lux[i, j, cl] = q[i, j, 0x2, c]
        luy[i, j, cl] = q[i, j, 0x3, c]
    end

    sync_threads()

    @inbounds begin
        dpijc_update = -zero(eltype(dq))
        duxijc_update = -zero(eltype(dq))
        duyijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, 0x1, c]

        wJijc = wJ[i, j, 0x1, c]

        #dpdt = -c ((r_x D_r + s_x D_s) ux + (r_y D_r + s_y D_s) uy)
        #duxdt = -c p_x = -c (r_x D_r + s_x D_s) p
        #duydt = -c p_y = -c (r_y D_r + s_y D_s) p

        for m = 0x1:N[1]
            dpijc_update -= dRdX[i, j, 0x1, c] * lDT1[i, m] * lux[m, j, cl] # -rₓDᵣuₓ
            dpijc_update -= dRdX[i, j, 0x3, c] * lDT1[i, m] * luy[m, j, cl] # -rᵥDᵣuᵥ

            duxijc_update -= dRdX[i, j, 0x1, c] * lDT1[i, m] * lp[m, j, cl]  # -rₓDᵣp
            duyijc_update -= dRdX[i, j, 0x3, c] * lDT1[i, m] * lp[m, j, cl]  # -rᵥDᵣp
        end

        for n = 0x1:N[2]
            dpijc_update -= dRdX[i, j, 0x2, c] * lDT2[j, n] * lux[i, n, cl] # -sₓDₛuₓ
            dpijc_update -= dRdX[i, j, 0x4, c] * lDT2[j, n] * luy[i, n, cl] # -sᵥDₛuᵥ

            duxijc_update -= dRdX[i, j, 0x2, c] * lDT2[j, n] * lp[i, n, cl]  # -sₓDₛp
            duyijc_update -= dRdX[i, j, 0x4, c] * lDT2[j, n] * lp[i, n, cl]  # -sᵥDₛp
        end

        dq[i, j, 0x1, c] += wJijc * invwJijc * dpijc_update
        dq[i, j, 0x2, c] += wJijc * invwJijc * duxijc_update
        dq[i, j, 0x3, c] += wJijc * invwJijc * duyijc_update
    end

    return nothing
end

function rhs_surface_kernel!(
    dq,
    q,
    vmapM,
    vmapP,
    mapB,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
) where {N,C}
    ij, cl = threadIdx()
    c = (blockIdx().x - 1) * blockDim().y + cl

    lpflux = CuStaticSharedArray(eltype(dq), (N..., C))
    luxflux = CuStaticSharedArray(eltype(dq), (N..., C))
    luyflux = CuStaticSharedArray(eltype(dq), (N..., C))

    @inbounds if ij <= N[1]
        i = ij
        for j = 1:N[2]
            lpflux[i, j, cl] = zero(eltype(lpflux))
            luxflux[i, j, cl] = zero(eltype(lpflux))
            luyflux[i, j, cl] = zero(eltype(lpflux))
        end
    end

    sync_threads()

    @inbounds if ij <= N[2]
        # Faces with r=-1 : West
        i = 1
        j = ij
        fid = j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = mapB[1, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp

        # Faces with r=1 : East 
        i = N[1]
        j = ij
        fid = N[2] + j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        idB = mapB[2, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp
    end

    sync_threads()

    @inbounds if ij <= N[1]
        # Faces with s=-1 : South 
        i = ij
        j = 1
        fid = 2N[2] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = mapB[3, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp

        i = ij
        j = N[2]
        fid = 2N[2] + N[1] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = mapB[0x4, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp
    end

    sync_threads()

    # RHS update
    i = ij
    @inbounds if i <= N[1]
        for j = 1:N[2]
            dq[i, j, 0x1, c] += lpflux[i, j, cl]
            dq[i, j, 0x2, c] += luxflux[i, j, cl]
            dq[i, j, 0x3, c] += luyflux[i, j, cl]
        end
    end
    return nothing
end

function rhs!(dq, q, grid, invwJ, DT, mapB, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    start!(q, cm)
    C = max(512 ÷ prod(size(cell)), 1)
    b = cld(last(size(dq)), C)

    @cuda threads = (size(cell)..., C) blocks = b rhs_volume_kernel!(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(cell)),
        Val(C),
    )


    finish!(q, cm)

    J = maximum(size(cell))
    C = max(128 ÷ J, 1)
    # J  x C workgroup sizes to evaluate  multiple elements on one wg.
    @cuda threads = (J, C) blocks = cld(last(size(dq)), C) rhs_surface_kernel!(
        parent(dq),
        parent(viewwithghosts(q)),
        fm.vmapM,
        fm.vmapP,
        mapB,
        parent(n),
        parent(wsJ),
        parent(invwJ),
        Val(size(cell)),
        Val(C),
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
    periodicity = (false, false)
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
                vtkfile["q"] = Adapt.adapt(Array, P * q)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    #jl # initialize state
    q = ic.(points(grid))
    dq = similar(q)
    dq .= Ref(zero(eltype(q)))

    #jl # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)
    #jl # precompute derivative transpose
    DT = derivatives_1d(cell)

    gridpoints_H = adapt(Array, grid.points)

    # adjust boundary code
    mapB_H = adapt(Array, boundarycodes(grid))
    for j = 1:size(mapB_H, 2)
        if last(gridpoints_H[end, end, j]) == 1  #North
            mapB_H[4, j] = 1
        end

        if last(gridpoints_H[1, 1, j]) == -1 # South
            mapB_H[3, j] = 1
        end

        if first(gridpoints_H[end, end, j]) == 1 # East 
            mapB_H[2, j] = 1
        end

        if first(gridpoints_H[1, 1, j]) == -1 # West 
            mapB_H[1, j] = 1
        end
    end

    mapB = adapt(AT, mapB_H)

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
            rhs!(dq, q, grid, invwJ, DT, mapB, cm)
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
    @assert CUDA.functional() && CUDA.has_cuda_gpu() "Nvidia GPU not available"
    AT = CuArray
    N = (7, 7)
    K = 4
    L = 4

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

    #jl # visualize solution
    vtkdir = "vtk_semidg_acoustic_2d$(K)x$(K)_L$(L)"
    if rank == 0
        @info """Starting Gaussian advection with:
            ($K, $K) coarse grid
            $L refinement level
        """
    end

    run(initialcondition, FT, AT, N, K, L; outputvtk = outputvtk, vtkdir, comm)
    rank == 0 && @info "Finished, vtk output written to $vtkdir"
end
