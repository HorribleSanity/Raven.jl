using WriteVTK: num_cells_structured
using CUDA: initialize_context
#--------------------------------Markdown Language Header-----------------------
# # 3D Acoustic Equation
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
using Pkg
using NVTX

const outputvtk = false 

function initialcondition(x::SVector{3})
    FT = eltype(x)
    p = exp(-(x[1]^2 + x[2]^2 + x[3]^2) / 0.8)
    uˣ = zero(FT)
    uʸ = zero(FT)
    uᶻ = zero(FT)
    return SVector{4,FT}(p, uˣ, uʸ, uᶻ)
end

# Accoustic box no open sides.
# n̂⋅u = 0 at ∂Ω and [[p]] = 0  accross ∂Ω
# pm = pp
# u⃗m = umtan + (n̂⋅u⃗m)n̂
# umta/ = u⃗m -(n̂⋅u⃗m)n̂
# u⃗p = umtan - (n̂⋅u⃗m)n̂ = u⃗m - 2(n̂⋅u⃗m)n̂

# BOUNDARY
# p* = α(pM + pP)/2 + |α|(1-α)[[pM - pP]]/2
# p* =  (pM + pP)/2 (central flux) note pM == pP
#    = pM
# dp = p* - pM = 0
# u* = α(uM + uP)/2 + |α|(1-α)[[uM - uP]]/2
#    =  (uM + uP)/2
#    =  (uM + uM - 2(n̂⋅u⃗M)n̂)/2
#    =  (uM - (n̂⋅u⃗M)n̂)
# du = uM - u* = (n̂⋅u⃗M)n̂
#
#
# internal edge w/ central flux
# dp = (pP + pM)/2 - pM = (pP - pM)/2
# du = (uP + uM)/2 - uM = (uP - uM)/2
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
    ijk, cl = threadIdx()
    c = (blockIdx().x - 1) * blockDim().y + cl

    pflux = CuStaticSharedArray(eltype(dq), (N..., C))
    uxflux = CuStaticSharedArray(eltype(dq), (N..., C))
    uyflux = CuStaticSharedArray(eltype(dq), (N..., C))
    uzflux = CuStaticSharedArray(eltype(dq), (N..., C))

    @inbounds if ijk <= N[1] * N[2]
        ij = ijk
        j, i = fldmod1(ij, N[1])

        for k = 1:N[3]
            z = zero(eltype(pflux))
            pflux[i, j, k, cl] = z
            uxflux[i, j, k, cl] = z
            uyflux[i, j, k, cl] = z
            uzflux[i, j, k, cl] = z
        end
    end

    sync_threads()

    @inbounds if ijk <= N[2] * N[3]
        # face with r = -1
        jk = ijk
        i = 1
        k, j = fldmod1(jk, N[2])

        fid = jk

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[1, c]     # global GPU memory access

        # n̂⋅u = 0 
        # pm = pp
        # u⃗m = umtan + (n̂⋅u⃗m)n̂
        # umtan = u⃗m - (n̂⋅u⃗m)n̂
        # u⃗p = umtan - (n̂⋅u⃗m)n̂ = u⃗m - 2(n̂⋅u⃗m)n̂
        #  
        # ⟹  2du⃗ = u⃗p - u⃗m = -2(n̂⋅u⃗)n̂
        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]

        if idB == 1
            dp = zero(eltype(pflux)) # pM == pP
            ndotu = nx * q[i, j, k, 2, c] + ny * q[i, j, k, 3, c] + nz * q[i, j, k, 4, c]
            dux = ndotu * nx
            duy = ndotu * ny
            duz = ndotu * nz
        else
            # indices of plus dof
            Pc, Pijk = fldmod1(idP, N[1] * N[2] * N[3])
            Pk, Pij = fldmod1(Pijk, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, k, 1, c] - q[Pi, Pj, Pk, 1, Pc]
            dux = q[i, j, k, 2, c] - q[Pi, Pj, Pk, 2, Pc]
            duy = q[i, j, k, 3, c] - q[Pi, Pj, Pk, 3, Pc]
            duz = q[i, j, k, 4, c] - q[Pi, Pj, Pk, 4, Pc]
        end

        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
        fscale = invwJijkc * wsJf / 2

        pflux[i, j, k, cl] += fscale * (nx * dux + ny * duy + nz * duz)
        uxflux[i, j, k, cl] += fscale * nx * dp
        uyflux[i, j, k, cl] += fscale * ny * dp
        uzflux[i, j, k, cl] += fscale * nz * dp

        # face with r = 1
        i = N[1]
        fid = N[2] * N[3] + jk

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[2, c]     # global GPU memory access

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]

        if idB == 1
            dp = zero(eltype(pflux)) # pM == pP
            ndotu = nx * q[i, j, k, 2, c] + ny * q[i, j, k, 3, c] + nz * q[i, j, k, 4, c]
            dux = ndotu * nx
            duy = ndotu * ny
            duz = ndotu * nz
        else
            Pc, Pijk = fldmod1(idP, N[1] * N[2] * N[3])
            Pk, Pij = fldmod1(Pijk, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, k, 1, c] - q[Pi, Pj, Pk, 1, Pc]
            dux = q[i, j, k, 2, c] - q[Pi, Pj, Pk, 2, Pc]
            duy = q[i, j, k, 3, c] - q[Pi, Pj, Pk, 3, Pc]
            duz = q[i, j, k, 4, c] - q[Pi, Pj, Pk, 4, Pc]
        end

        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
        fscale = invwJijkc * wsJf / 2


        pflux[i, j, k, cl] += fscale * (nx * dux + ny * duy + nz * duz)
        uxflux[i, j, k, cl] += fscale * nx * dp
        uyflux[i, j, k, cl] += fscale * ny * dp
        uzflux[i, j, k, cl] += fscale * nz * dp

    end

    sync_threads()

    @inbounds if ijk <= N[1] * N[3]
        # face with s = -1
        ik = ijk
        j = 1
        k, i = fldmod1(ik, N[1])

        fid = 2 * N[2] * N[3] + ik

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[3, c]     # global GPU memory access

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]

        if idB == 1
            dp = zero(eltype(pflux)) # pM == pP
            ndotu = nx * q[i, j, k, 2, c] + ny * q[i, j, k, 3, c] + nz * q[i, j, k, 4, c]

            dux = ndotu * nx
            duy = ndotu * ny
            duz = ndotu * nz
        else
            Pc, Pijk = fldmod1(idP, N[1] * N[2] * N[3])
            Pk, Pij = fldmod1(Pijk, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, k, 1, c] - q[Pi, Pj, Pk, 1, Pc]
            dux = q[i, j, k, 2, c] - q[Pi, Pj, Pk, 2, Pc]
            duy = q[i, j, k, 3, c] - q[Pi, Pj, Pk, 3, Pc]
            duz = q[i, j, k, 4, c] - q[Pi, Pj, Pk, 4, Pc]
        end

        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
        fscale = invwJijkc * wsJf / 2

        pflux[i, j, k, cl] += fscale * (nx * dux + ny * duy + nz * duz)
        uxflux[i, j, k, cl] += fscale * nx * dp
        uyflux[i, j, k, cl] += fscale * ny * dp
        uzflux[i, j, k, cl] += fscale * nz * dp

        # face with s = 1
        j = N[2]
        fid = 2 * N[2] * N[3] + N[1] * N[3] + ik

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[4, c]     # global GPU memory access

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]

        if idB == 1
            dp = zero(eltype(pflux)) # pM == pP
            ndotu = nx * q[i, j, k, 2, c] + ny * q[i, j, k, 3, c] + nz * q[i, j, k, 4, c]

            dux = ndotu * nx
            duy = ndotu * ny
            duz = ndotu * nz
        else
            Pc, Pijk = fldmod1(idP, N[1] * N[2] * N[3])
            Pk, Pij = fldmod1(Pijk, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, k, 1, c] - q[Pi, Pj, Pk, 1, Pc]
            dux = q[i, j, k, 2, c] - q[Pi, Pj, Pk, 2, Pc]
            duy = q[i, j, k, 3, c] - q[Pi, Pj, Pk, 3, Pc]
            duz = q[i, j, k, 4, c] - q[Pi, Pj, Pk, 4, Pc]
        end

        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
        fscale = invwJijkc * wsJf / 2


        pflux[i, j, k, cl] += fscale * (nx * dux + ny * duy + nz * duz)
        uxflux[i, j, k, cl] += fscale * nx * dp
        uyflux[i, j, k, cl] += fscale * ny * dp
        uzflux[i, j, k, cl] += fscale * nz * dp
    end

    sync_threads()

    @inbounds if ijk <= N[1] * N[2]
        # face with t = -1
        ij = ijk

        j, i = fldmod1(ij, N[1])
        k = 1

        fid = 2 * (N[2] * N[3] + N[1] * N[3]) + ij

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[5, c]     # global GPU memory access

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]

        if idB == 1
            dp = zero(eltype(pflux)) # pM == pP
            ndotu = nx * q[i, j, k, 2, c] + ny * q[i, j, k, 3, c] + nz * q[i, j, k, 4, c]

            dux = ndotu * nx
            duy = ndotu * ny
            duz = ndotu * nz
        else
            Pc, Pijk = fldmod1(idP, N[1] * N[2] * N[3])
            Pk, Pij = fldmod1(Pijk, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, k, 1, c] - q[Pi, Pj, Pk, 1, Pc]
            dux = q[i, j, k, 2, c] - q[Pi, Pj, Pk, 2, Pc]
            duy = q[i, j, k, 3, c] - q[Pi, Pj, Pk, 3, Pc]
            duz = q[i, j, k, 4, c] - q[Pi, Pj, Pk, 4, Pc]
        end

        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
        fscale = invwJijkc * wsJf / 2


        pflux[i, j, k, cl] += fscale * (nx * dux + ny * duy + nz * duz)
        uxflux[i, j, k, cl] += fscale * nx * dp
        uyflux[i, j, k, cl] += fscale * ny * dp
        uzflux[i, j, k, cl] += fscale * nz * dp

        # face with t = 1
        k = N[3]
        fid = 2 * (N[2] * N[3] + N[1] * N[3]) + N[1] * N[2] + ij

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[6, c]     # global GPU memory access

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]

        if idB == 1
            dp = zero(eltype(pflux)) # pM == pP
            ndotu = nx * q[i, j, k, 2, c] + ny * q[i, j, k, 3, c] + nz * q[i, j, k, 4, c]

            dux = ndotu * nx
            duy = ndotu * ny
            duz = ndotu * nz
        else
            Pc, Pijk = fldmod1(idP, N[1] * N[2] * N[3])
            Pk, Pij = fldmod1(Pijk, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, k, 1, c] - q[Pi, Pj, Pk, 1, Pc]
            dux = q[i, j, k, 2, c] - q[Pi, Pj, Pk, 2, Pc]
            duy = q[i, j, k, 3, c] - q[Pi, Pj, Pk, 3, Pc]
            duz = q[i, j, k, 4, c] - q[Pi, Pj, Pk, 4, Pc]
        end

        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
        fscale = invwJijkc * wsJf / 2


        pflux[i, j, k, cl] += fscale * (nx * dux + ny * duy + nz * duz)
        uxflux[i, j, k, cl] += fscale * nx * dp
        uyflux[i, j, k, cl] += fscale * ny * dp
        uzflux[i, j, k, cl] += fscale * nz * dp
    end

    sync_threads()

    ij = ijk
    @inbounds if ij <= N[1] * N[2]
        j, i = fldmod1(ij, N[1])

        for k = 1:N[3]
            dq[i, j, k, 1, c] += pflux[i, j, k, cl]
            dq[i, j, k, 2, c] += uxflux[i, j, k, cl]
            dq[i, j, k, 3, c] += uyflux[i, j, k, cl]
            dq[i, j, k, 4, c] += uzflux[i, j, k, cl]
        end
    end
    return nothing
end

function rhs_volume_kernel!(dq, q, dRdX, wJ, invwJ, DT, ::Val{N}, ::Val{C}) where {N,C}
    i, j, k = threadIdx()
    cl = 0x1
    c = blockIdx().x

    lDT1 = CuStaticSharedArray(Float64, (N[1], N[1]))
    lDT2 = CuStaticSharedArray(Float64, (N[2], N[2]))
    lDT3 = CuStaticSharedArray(Float64, (N[3], N[3]))

    lp = CuStaticSharedArray(Float64, (N..., C))
    luˣ = CuStaticSharedArray(Float64, (N..., C))
    luʸ = CuStaticSharedArray(Float64, (N..., C))
    luᶻ = CuStaticSharedArray(Float64, (N..., C))

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

        for si = 0x0:N[1]:(N[3]-0x1)
            if i + si <= N[3] && cl == 0x1
                lDT3[i+si, k] = DT[3][i+si, k]
            end
        end

        # loading into local mem data (i, j, k) from global index cell c
        lp[i, j, k, cl] = q[i, j, k, 0x1, c]
        luˣ[i, j, k, cl] = q[i, j, k, 0x2, c]
        luʸ[i, j, k, cl] = q[i, j, k, 0x3, c]
        luᶻ[i, j, k, cl] = q[i, j, k, 0x4, c]
    end

    sync_threads()

    @inbounds begin
        dpijkc_update = -zero(eltype(dq))
        duxijkc_update = -zero(eltype(dq))
        duyijkc_update = -zero(eltype(dq))
        duzijkc_update = -zero(eltype(dq))

        invwJijkc = invwJ[i, j, k, 1, c]
        wJijkc = wJ[i, j, k, 1, c]

        #dpdt = -c ((r_x D_r + s_x D_s+ t_x D_t) ux 
        #         + (r_y D_r + s_y D_s+ t_y D_t) uy 
        #         + (r_z D_r + s_z D_s+ t_z D_t) uz)
        #duxdt = -c p_x = -c (r_x D_r + s_x D_s+ t_x D_t) p
        #duydt = -c p_y = -c (r_y D_r + s_y D_s+ t_y D_t) p
        #duzdt = -c p_z = -c (r_z D_r + s_z D_s+ t_z D_t) p

        for l = 0x1:N[1]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 1, c] * lDT1[i, l] * luˣ[l, j, k, cl]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 4, c] * lDT1[i, l] * luʸ[l, j, k, cl]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 7, c] * lDT1[i, l] * luᶻ[l, j, k, cl]

            duxijkc_update -= wJijkc * dRdX[i, j, k, 1, c] * lDT1[i, l] * lp[l, j, k, cl]
            duyijkc_update -= wJijkc * dRdX[i, j, k, 4, c] * lDT1[i, l] * lp[l, j, k, cl]
            duzijkc_update -= wJijkc * dRdX[i, j, k, 7, c] * lDT1[i, l] * lp[l, j, k, cl]
        end

        sync_threads()

        for n = 0x1:N[2]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 2, c] * lDT2[j, n] * luˣ[i, n, k, cl]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 5, c] * lDT2[j, n] * luʸ[i, n, k, cl]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 8, c] * lDT2[j, n] * luᶻ[i, n, k, cl]

            duxijkc_update -= wJijkc * dRdX[i, j, k, 2, c] * lDT2[j, n] * lp[i, n, k, cl]
            duyijkc_update -= wJijkc * dRdX[i, j, k, 5, c] * lDT2[j, n] * lp[i, n, k, cl]
            duzijkc_update -= wJijkc * dRdX[i, j, k, 8, c] * lDT2[j, n] * lp[i, n, k, cl]
        end

        sync_threads()

        for m = 0x1:N[3]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * luˣ[i, j, m, cl]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * luʸ[i, j, m, cl]
            dpijkc_update -= wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * luᶻ[i, j, m, cl]

            duxijkc_update -= wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lp[i, j, m, cl]
            duyijkc_update -= wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lp[i, j, m, cl]
            duzijkc_update -= wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lp[i, j, m, cl]
        end

        sync_threads()

        dq[i, j, k, 0x1, c] += invwJijkc * dpijkc_update
        dq[i, j, k, 0x2, c] += invwJijkc * duxijkc_update
        dq[i, j, k, 0x3, c] += invwJijkc * duyijkc_update
        dq[i, j, k, 0x4, c] += invwJijkc * duzijkc_update
    end
    return nothing
end

function rhs!(dq, q, grid, invwJ, DT, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    start!(q, cm)

    C = 1 #max(512 ÷ prod(size(cell)), 1)
    b = cld(size(dq)[4], C)

    @cuda threads = size(cell) blocks = b rhs_volume_kernel!(
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

    J = maximum([prod(size(cell)[[idx...]]) for idx in [(1, 2), (2, 3), (1, 3)]])
    C = 2^3# max(128 ÷ J, 1)

    @cuda threads = (J, C) blocks = cld(last(size(dq)), C) rhs_surface_kernel!(
        parent(dq),
        parent(viewwithghosts(q)),
        fm.vmapM,
        fm.vmapP,
        boundarycodes(grid),
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
    coordinates = ntuple(_ -> range(FT(-1), stop = FT(1), length = K + 1), 3)
    periodicity = (false, false, false)
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(gm)

    timeend = 2.5

    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates))
    dt = cfl * dx / (maximum(N))^3

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
        cd(vtkdir) do
            filename = "step$(lpad(step, 6, '0'))"
            vtkfile = vtk_grid(filename, grid)
            P = toequallyspaced(cell)
            vtkfile["p"] = Adapt.adapt(Array, P * getindex.(q, 1))
            vtkfile["uˣ"] = Adapt.adapt(Array, P * getindex.(q, 2))
            vtkfile["uʸ"] = Adapt.adapt(Array, P * getindex.(q, 3))
            vtkfile["uᶻ"] = Adapt.adapt(Array, P * getindex.(q, 4))
            vtk_save(vtkfile)
            if rank == 0
                pvd[time] = vtkfile
            end
        end
    end

    q = ic.(points(grid))
    dq = similar(q)
    dq .= Ref(zero(eltype(q)))
    #jl # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)
    #jl # precompute derivative transpose
    DT = derivatives_1d(cell)

    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #jl # initial output
    step = 0
    time = FT(0)

    if outputvtk
        do_output(step, time, q)
    end

    for step = 1:numberofsteps
        if time + dt > timeend
            dt = timeend - time
        end

        for (i, stage) in enumerate(eachindex(RKA, RKB))
            @. dq *= RKA[stage]
            rhs!(dq, q, grid, invwJ, DT, cm)
            @. q += RKB[stage] * dt * dq
        end
        time += dt

        if outputvtk && step % 100 == 0
            do_output(step, time, q)
        end
    end

    #final output
    if outputvtk
        do_output(numberofsteps, timeend, q)
        if rank == 0
            cd(vtkdir) do
                vtk_save(pvd)
            end
        end
    end

    return
end

let
    FT = Float64
    @assert CUDA.functional() && CUDA.has_cuda_gpu() "NVidia GPU not available"
    AT = CuArray

    N = (4, 4, 4)

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

    K = 6
    L = 2
    vtkdir = "vtk_semdg_acoustic_3d$(K)x$(K)x$(K)_L$(L)"
    if rank == 0
        @info """Starting Acoustic test problem with:
            ($K, $K, $K) coarse grid
            $L refinement level
        """
    end

    run(initialcondition, FT, AT, N, K, L; outputvtk = outputvtk, vtkdir, comm)
    if outputvtk && rank == 0 && @info "Finished, vtk output written to $vtkdir"
    end
end
