#--------------------------------Markdown Language Header-----------------------
# # 3D Maxwells equation
#--------------------------------Markdown Language Header-----------------------
using WriteVTK: num_cells_structured
using CUDA: initialize_context
using Base: sign_mask
using Adapt
using MPI
using CUDA
using LinearAlgebra
using Printf
using Raven
using Raven.KernelAbstractions.Extras: @unroll
using StaticArrays
using WriteVTK
using Pkg

using ProgressBars

const outputvtk = false 
const convergetest = false 


function solution(x::SVector{3}, t)
    FT = eltype(x)
    m = n = 2
    ω = π*sqrt(m^2+n^2)
    Hˣ = -(π*n/ω)*sin(m*π*x[1])*cos(n*π*x[2])*sin(ω*t)
    Hʸ =  (π*m/ω)*cos(m*π*x[1])*sin(n*π*x[2])*sin(ω*t)
    Eᶻ =          sin(m*π*x[1])*sin(n*π*x[2])*cos(ω*t)
    z = zero(FT)
    return SVector{6,FT}(Hˣ,Hʸ,z,z,z,Eᶻ)
end

initialcondition(x::SVector{3}) = solution(x,0.0)

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
    ::Val{C}
) where {N, C}
    ijk, cl = threadIdx()
    c = (blockIdx().x-0x1)*C + cl

    Hxflux = CuStaticSharedArray(eltype(dq), (N..., C))
    Hyflux = CuStaticSharedArray(eltype(dq), (N..., C))
    Hzflux = CuStaticSharedArray(eltype(dq), (N..., C))

    Exflux = CuStaticSharedArray(eltype(dq), (N..., C))
    Eyflux = CuStaticSharedArray(eltype(dq), (N..., C))
    Ezflux = CuStaticSharedArray(eltype(dq), (N..., C))

    @inbounds if ijk <= N[0x1] * N[0x2]
        ij = ijk
        j, i = fldmod1(ij, N[0x1])

        for k = 1:N[0x3]
            z = zero(eltype(Hxflux))
            Hxflux[i, j, k, cl] = z
            Hyflux[i, j, k, cl] = z
            Hzflux[i, j, k, cl] = z

            Exflux[i, j, k, cl] = z
            Eyflux[i, j, k, cl] = z
            Ezflux[i, j, k, cl] = z
        end
    end

    sync_threads()

    @inbounds if ijk <= N[0x2]*N[0x3]
        alpha = 1.0
        # face with r = -1
        jk = ijk
        i = 0x1
        k, j = fldmod1(jk, N[0x2])

        fid = jk 

        idP = vmapP[fid, c]
        idB = mapB[0x1, c]

        if idB == 0x1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 0x4, c]
            dEy = -2 * q[i, j, k, 0x5, c]
            dEz = -2 * q[i, j, k, 0x6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[0x1]*N[0x2]*N[0x3])
            Pk, Pij  = fldmod1(Pijk, N[0x1]*N[0x2])
            Pj, Pi   = fldmod1(Pij,  N[0x1])

            dHx = q[Pi, Pj, Pk, 0x1, Pc] - q[i, j, k, 0x1, c]
            dHy = q[Pi, Pj, Pk, 0x2, Pc] - q[i, j, k, 0x2, c]
            dHz = q[Pi, Pj, Pk, 0x3, Pc] - q[i, j, k, 0x3, c]

            dEx = q[Pi, Pj, Pk, 0x4, Pc] - q[i, j, k, 0x4, c]
            dEy = q[Pi, Pj, Pk, 0x5, Pc] - q[i, j, k, 0x5, c]
            dEz = q[Pi, Pj, Pk, 0x6, Pc] - q[i, j, k, 0x6, c]
        end

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]
        nz = n[0x3, fid, c]
        wsJf = wsJ[0x1, fid, c]

        invwJijkc = invwJ[i, j, k, 0x1, c]
        fscale = invwJijkc * wsJf / 2

        ndotdH = nx*dHx + ny*dHy + nz*dHz
        ndotdE = nx*dEx + ny*dEy + nz*dEz

        Hxflux[i, j, k, cl] += fscale * (-ny * dEz + nz*dEy + alpha*(dHx - ndotdH*nx))
        Hyflux[i, j, k, cl] += fscale * (-nz * dEx + nx*dEz + alpha*(dHy - ndotdH*ny))
        Hzflux[i, j, k, cl] += fscale * (-nx * dEy + ny*dEx + alpha*(dHz - ndotdH*nz))

        Exflux[i, j, k, cl] += fscale * ( ny * dHz - nz*dHy + alpha*(dEx - ndotdE*nx))
        Eyflux[i, j, k, cl] += fscale * ( nz * dHx - nx*dHz + alpha*(dEy - ndotdE*ny))
        Ezflux[i, j, k, cl] += fscale * ( nx * dHy - ny*dHx + alpha*(dEz - ndotdE*nz))

        # face with r = 1
        i = N[0x1]
        fid = N[0x2]*N[0x3] + jk

        idP = vmapP[fid, c]
        idB = mapB[0x2, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 0x4, c]
            dEy = -2 * q[i, j, k, 0x5, c]
            dEz = -2 * q[i, j, k, 0x6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[0x1]*N[0x2]*N[0x3])
            Pk, Pij  = fldmod1(Pijk, N[0x1]*N[0x2])
            Pj, Pi   = fldmod1(Pij,  N[0x1])

            dHx = q[Pi, Pj, Pk, 0x1, Pc] - q[i, j, k, 0x1, c]
            dHy = q[Pi, Pj, Pk, 0x2, Pc] - q[i, j, k, 0x2, c]
            dHz = q[Pi, Pj, Pk, 0x3, Pc] - q[i, j, k, 0x3, c]

            dEx = q[Pi, Pj, Pk, 0x4, Pc] - q[i, j, k, 0x4, c]
            dEy = q[Pi, Pj, Pk, 0x5, Pc] - q[i, j, k, 0x5, c]
            dEz = q[Pi, Pj, Pk, 0x6, Pc] - q[i, j, k, 0x6, c]

        end

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]
        nz = n[0x3, fid, c]
        wsJf = wsJ[0x1, fid, c]

        invwJijkc = invwJ[i, j, k, 0x1, c]
        fscale = invwJijkc * wsJf / 2

        ndotdH = nx*dHx + ny*dHy + nz*dHz
        ndotdE = nx*dEx + ny*dEy + nz*dEz

        Hxflux[i, j, k, cl] += fscale * (-ny * dEz + nz*dEy + alpha*(dHx - ndotdH*nx))
        Hyflux[i, j, k, cl] += fscale * (-nz * dEx + nx*dEz + alpha*(dHy - ndotdH*ny))
        Hzflux[i, j, k, cl] += fscale * (-nx * dEy + ny*dEx + alpha*(dHz - ndotdH*nz))

        Exflux[i, j, k, cl] += fscale * ( ny * dHz - nz*dHy + alpha*(dEx - ndotdE*nx))
        Eyflux[i, j, k, cl] += fscale * ( nz * dHx - nx*dHz + alpha*(dEy - ndotdE*ny)) 
        Ezflux[i, j, k, cl] += fscale * ( nx * dHy - ny*dHx + alpha*(dEz - ndotdE*nz)) 
    end

    sync_threads()

    @inbounds if ijk <= N[0x1]*N[0x3]
        alpha = 1.0
        # face with s = -1
        ik = ijk
        j = 1
        k, i = fldmod1(ik, N[0x1])

        fid = 2 * N[0x2] * N[0x3] + ik

        idP = vmapP[fid, c]
        idB = mapB[0x3, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 0x4, c]
            dEy = -2 * q[i, j, k, 0x5, c]
            dEz = -2 * q[i, j, k, 0x6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[0x1]*N[0x2]*N[0x3])
            Pk, Pij  = fldmod1(Pijk, N[0x1]*N[0x2])
            Pj, Pi   = fldmod1(Pij,  N[0x1])

            dHx = q[Pi, Pj, Pk, 0x1, Pc] - q[i, j, k, 0x1, c]
            dHy = q[Pi, Pj, Pk, 0x2, Pc] - q[i, j, k, 0x2, c]
            dHz = q[Pi, Pj, Pk, 0x3, Pc] - q[i, j, k, 0x3, c]

            dEx = q[Pi, Pj, Pk, 0x4, Pc] - q[i, j, k, 0x4, c]
            dEy = q[Pi, Pj, Pk, 0x5, Pc] - q[i, j, k, 0x5, c]
            dEz = q[Pi, Pj, Pk, 0x6, Pc] - q[i, j, k, 0x6, c]

        end

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]
        nz = n[0x3, fid, c]
        wsJf = wsJ[0x1, fid, c]

        invwJijkc = invwJ[i, j, k, 0x1, c]
        fscale = invwJijkc * wsJf / 2

        ndotdH = nx*dHx + ny*dHy + nz*dHz
        ndotdE = nx*dEx + ny*dEy + nz*dEz

        Hxflux[i, j, k, cl] += fscale * (-ny * dEz + nz*dEy + alpha*(dHx - ndotdH*nx))
        Hyflux[i, j, k, cl] += fscale * (-nz * dEx + nx*dEz + alpha*(dHy - ndotdH*ny))
        Hzflux[i, j, k, cl] += fscale * (-nx * dEy + ny*dEx + alpha*(dHz - ndotdH*nz))

        Exflux[i, j, k, cl] += fscale * ( ny * dHz - nz*dHy + alpha*(dEx - ndotdE*nx))
        Eyflux[i, j, k, cl] += fscale * ( nz * dHx - nx*dHz + alpha*(dEy - ndotdE*ny))
        Ezflux[i, j, k, cl] += fscale * ( nx * dHy - ny*dHx + alpha*(dEz - ndotdE*nz))

        # face with s = 1
        j = N[0x2]
        fid = 2*N[0x2]*N[0x3] + N[0x1]*N[0x3]  + ik

        idP = vmapP[fid, c]
        idB = mapB[0x4, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 0x4, c]
            dEy = -2 * q[i, j, k, 0x5, c]
            dEz = -2 * q[i, j, k, 0x6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[0x1]*N[0x2]*N[0x3])
            Pk, Pij  = fldmod1(Pijk, N[0x1]*N[0x2])
            Pj, Pi   = fldmod1(Pij,  N[0x1])

            dHx = q[Pi, Pj, Pk, 0x1, Pc] - q[i, j, k, 0x1, c]
            dHy = q[Pi, Pj, Pk, 0x2, Pc] - q[i, j, k, 0x2, c]
            dHz = q[Pi, Pj, Pk, 0x3, Pc] - q[i, j, k, 0x3, c]

            dEx = q[Pi, Pj, Pk, 0x4, Pc] - q[i, j, k, 0x4, c]
            dEy = q[Pi, Pj, Pk, 0x5, Pc] - q[i, j, k, 0x5, c]
            dEz = q[Pi, Pj, Pk, 0x6, Pc] - q[i, j, k, 0x6, c]

        end

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]
        nz = n[0x3, fid, c]
        wsJf = wsJ[0x1, fid, c]

        invwJijkc = invwJ[i, j, k, 0x1, c]
        fscale = invwJijkc * wsJf / 2

        ndotdH = nx*dHx + ny*dHy + nz*dHz
        ndotdE = nx*dEx + ny*dEy + nz*dEz

        Hxflux[i, j, k, cl] += fscale * (-ny * dEz + nz*dEy + alpha*(dHx - ndotdH*nx))
        Hyflux[i, j, k, cl] += fscale * (-nz * dEx + nx*dEz + alpha*(dHy - ndotdH*ny))
        Hzflux[i, j, k, cl] += fscale * (-nx * dEy + ny*dEx + alpha*(dHz - ndotdH*nz))

        Exflux[i, j, k, cl] += fscale * ( ny * dHz - nz*dHy + alpha*(dEx - ndotdE*nx))
        Eyflux[i, j, k, cl] += fscale * ( nz * dHx - nx*dHz + alpha*(dEy - ndotdE*ny))
        Ezflux[i, j, k, cl] += fscale * ( nx * dHy - ny*dHx + alpha*(dEz - ndotdE*nz))
    end


    sync_threads()

    @inbounds if ijk <= N[0x1]*N[0x2]
        alpha = 1.0
        # face with t = -1
        ij = ijk

        j, i = fldmod1(ij, N[0x1])
        k = 1

        fid = 2 * (N[0x2] * N[0x3] + N[0x1] * N[0x3]) + ij 

        idP = vmapP[fid, c]
        idB = mapB[0x5, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 0x4, c]
            dEy = -2 * q[i, j, k, 0x5, c]
            dEz = -2 * q[i, j, k, 0x6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[0x1]*N[0x2]*N[0x3])
            Pk, Pij  = fldmod1(Pijk, N[0x1]*N[0x2])
            Pj, Pi   = fldmod1(Pij,  N[0x1])

            dHx = q[Pi, Pj, Pk, 0x1, Pc] - q[i, j, k, 0x1, c]
            dHy = q[Pi, Pj, Pk, 0x2, Pc] - q[i, j, k, 0x2, c]
            dHz = q[Pi, Pj, Pk, 0x3, Pc] - q[i, j, k, 0x3, c]

            dEx = q[Pi, Pj, Pk, 0x4, Pc] - q[i, j, k, 0x4, c]
            dEy = q[Pi, Pj, Pk, 0x5, Pc] - q[i, j, k, 0x5, c]
            dEz = q[Pi, Pj, Pk, 0x6, Pc] - q[i, j, k, 0x6, c]
        end

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]
        nz = n[0x3, fid, c]
        wsJf = wsJ[0x1, fid, c]

        invwJijkc = invwJ[i, j, k, 0x1, c]
        fscale = invwJijkc * wsJf / 2

        ndotdH = nx*dHx + ny*dHy + nz*dHz
        ndotdE = nx*dEx + ny*dEy + nz*dEz

        Hxflux[i, j, k, cl] += fscale * (-ny * dEz + nz*dEy + alpha*(dHx - ndotdH*nx))
        Hyflux[i, j, k, cl] += fscale * (-nz * dEx + nx*dEz + alpha*(dHy - ndotdH*ny))
        Hzflux[i, j, k, cl] += fscale * (-nx * dEy + ny*dEx + alpha*(dHz - ndotdH*nz))

        Exflux[i, j, k, cl] += fscale * ( ny * dHz - nz*dHy + alpha*(dEx - ndotdE*nx))
        Eyflux[i, j, k, cl] += fscale * ( nz * dHx - nx*dHz + alpha*(dEy - ndotdE*ny))
        Ezflux[i, j, k, cl] += fscale * ( nx * dHy - ny*dHx + alpha*(dEz - ndotdE*nz))

        # face with t = 1
        k = N[0x3]
        fid = 2 * (N[0x2] * N[0x3] + N[0x1] * N[0x3]) + N[0x1] * N[0x2]  + ij

        idP = vmapP[fid, c]
        idB = mapB[0x6, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 0x4, c]
            dEy = -2 * q[i, j, k, 0x5, c]
            dEz = -2 * q[i, j, k, 0x6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[0x1]*N[0x2]*N[0x3])
            Pk, Pij  = fldmod1(Pijk, N[0x1]*N[0x2])
            Pj, Pi   = fldmod1(Pij,  N[0x1])

            dHx = q[Pi, Pj, Pk, 0x1, Pc] - q[i, j, k, 0x1, c]
            dHy = q[Pi, Pj, Pk, 0x2, Pc] - q[i, j, k, 0x2, c]
            dHz = q[Pi, Pj, Pk, 0x3, Pc] - q[i, j, k, 0x3, c]

            dEx = q[Pi, Pj, Pk, 0x4, Pc] - q[i, j, k, 0x4, c]
            dEy = q[Pi, Pj, Pk, 0x5, Pc] - q[i, j, k, 0x5, c]
            dEz = q[Pi, Pj, Pk, 0x6, Pc] - q[i, j, k, 0x6, c]
        end

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]
        nz = n[0x3, fid, c]
        wsJf = wsJ[0x1, fid, c]

        invwJijkc = invwJ[i, j, k, 0x1, c]
        fscale = invwJijkc * wsJf / 2

        ndotdH = nx*dHx + ny*dHy + nz*dHz
        ndotdE = nx*dEx + ny*dEy + nz*dEz

        Hxflux[i, j, k, cl] += fscale * (-ny * dEz + nz*dEy + alpha*(dHx - ndotdH*nx))
        Hyflux[i, j, k, cl] += fscale * (-nz * dEx + nx*dEz + alpha*(dHy - ndotdH*ny))
        Hzflux[i, j, k, cl] += fscale * (-nx * dEy + ny*dEx + alpha*(dHz - ndotdH*nz))

        Exflux[i, j, k, cl] += fscale * ( ny * dHz - nz*dHy + alpha*(dEx - ndotdE*nx))
        Eyflux[i, j, k, cl] += fscale * ( nz * dHx - nx*dHz + alpha*(dEy - ndotdE*ny))
        Ezflux[i, j, k, cl] += fscale * ( nx * dHy - ny*dHx + alpha*(dEz - ndotdE*nz))
    end

    sync_threads()

    ij = ijk
    @inbounds if ij <= N[0x1] * N[0x2]
        j, i = fldmod1(ij, N[0x1])

        for k = 1:N[0x3]
            dq[i, j, k, 0x1, c] += Hxflux[i, j, k, cl]
            dq[i, j, k, 0x2, c] += Hyflux[i, j, k, cl]
            dq[i, j, k, 0x3, c] += Hzflux[i, j, k, cl]
            dq[i, j, k, 0x4, c] += Exflux[i, j, k, cl]
            dq[i, j, k, 0x5, c] += Eyflux[i, j, k, cl]
            dq[i, j, k, 0x6, c] += Ezflux[i, j, k, cl]
        end
    end
    return nothing
end

function rhs_volume_vertical_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    ::Val{G},
    ::Val{N},
    ::Val{ISTRIDE},
) where {G, N, ISTRIDE}
    il, j, k = threadIdx()
    c, iblockidx = blockIdx()
    i = (iblockidx-0x1)*ISTRIDE + il

    lDT3 = CuStaticSharedArray(eltype(dq), (N[0x3], N[0x3]))

    lHˣ = CuStaticSharedArray(eltype(dq), (ISTRIDE, N[0x2], N[0x3]))
    lHʸ = CuStaticSharedArray(eltype(dq), (ISTRIDE, N[0x2], N[0x3]))
    lHᶻ = CuStaticSharedArray(eltype(dq), (ISTRIDE, N[0x2], N[0x3]))

    lEˣ = CuStaticSharedArray(eltype(dq), (ISTRIDE, N[0x2], N[0x3]))
    lEʸ = CuStaticSharedArray(eltype(dq), (ISTRIDE, N[0x2], N[0x3]))
    lEᶻ = CuStaticSharedArray(eltype(dq), (ISTRIDE, N[0x2], N[0x3]))

    @inbounds if i <= G[0x1]
        for sj = 0x0:N[0x2]:(N[0x3]-0x1)
            if j+sj <= N[0x3] && il == 1
                lDT3[j+sj, k] = DT[0x3][j+sj, k]
            end
        end
    
        lHˣ[il, j, k] = q[i, j, k, 0x1, c]
        lHʸ[il, j, k] = q[i, j, k, 0x2, c]
        lHᶻ[il, j, k] = q[i, j, k, 0x3, c]
        lEˣ[il, j, k] = q[i, j, k, 0x4, c]
        lEʸ[il, j, k] = q[i, j, k, 0x5, c]
        lEᶻ[il, j, k] = q[i, j, k, 0x6, c]
    end

    sync_threads()

    @inbounds  if i <= G[0x1]
        dHˣijkc_update = -zero(eltype(dq))
        dHʸijkc_update = -zero(eltype(dq))
        dHᶻijkc_update = -zero(eltype(dq))
        dEˣijkc_update = -zero(eltype(dq))
        dEʸijkc_update = -zero(eltype(dq))
        dEᶻijkc_update = -zero(eltype(dq))

        invwJijkc = invwJ[i, j, k, 0x1, c]
        wJijkc = wJ[i, j, k, 0x1, c]

        wJdRdXijkc_3 = wJijkc * dRdX[i, j, k, 0x3, c]
        wJdRdXijkc_6 = wJijkc * dRdX[i, j, k, 0x6, c]
        wJdRdXijkc_9 = wJijkc * dRdX[i, j, k, 0x9, c]

        @unroll for m = 0x1:N[0x3]
            lDT3km = lDT3[k, m]
            dHˣijkc_update -= wJdRdXijkc_6 * lDT3km * lEᶻ[il, j, m]
            dHˣijkc_update += wJdRdXijkc_9 * lDT3km * lEʸ[il, j, m]

            dHʸijkc_update -= wJdRdXijkc_9 * lDT3km * lEˣ[il, j, m]
            dHʸijkc_update += wJdRdXijkc_3 * lDT3km * lEᶻ[il, j, m]

            dHᶻijkc_update -= wJdRdXijkc_3 * lDT3km * lEʸ[il, j, m]
            dHᶻijkc_update += wJdRdXijkc_6 * lDT3km * lEˣ[il, j, m]

            dEˣijkc_update += wJdRdXijkc_6 * lDT3km * lHᶻ[il, j, m]
            dEˣijkc_update -= wJdRdXijkc_9 * lDT3km * lHʸ[il, j, m]
            dEʸijkc_update += wJdRdXijkc_9 * lDT3km * lHˣ[il, j, m]
            dEʸijkc_update -= wJdRdXijkc_3 * lDT3km * lHᶻ[il, j, m]

            dEᶻijkc_update += wJdRdXijkc_3 * lDT3km * lHʸ[il, j, m]
            dEᶻijkc_update -= wJdRdXijkc_6 * lDT3km * lHˣ[il, j, m]
        end

        dq[i, j, k, 0x1, c] += invwJijkc * dHˣijkc_update
        dq[i, j, k, 0x2, c] += invwJijkc * dHʸijkc_update
        dq[i, j, k, 0x3, c] += invwJijkc * dHᶻijkc_update
        dq[i, j, k, 0x4, c] += invwJijkc * dEˣijkc_update
        dq[i, j, k, 0x5, c] += invwJijkc * dEʸijkc_update
        dq[i, j, k, 0x6, c] += invwJijkc * dEᶻijkc_update
    end
    return nothing
end

function rhs_volume_horizontal_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    ::Val{G},
    ::Val{N},
    ::Val{KSTRIDE},
) where {G, N, KSTRIDE}
    i, j, kl = threadIdx()
    c, kblockidx = blockIdx()
    k = (kblockidx - 0x1) * KSTRIDE + kl

    lDT1 = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x1]))
    lDT2 = CuStaticSharedArray(eltype(dq), (N[0x2], N[0x2]))

    lHˣ = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x2], KSTRIDE))
    lHʸ = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x2], KSTRIDE))
    lHᶻ = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x2], KSTRIDE))

    lEˣ = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x2], KSTRIDE))
    lEʸ = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x2], KSTRIDE))
    lEᶻ = CuStaticSharedArray(eltype(dq), (N[0x1], N[0x2], KSTRIDE))

    @inbounds if k <= G[0x3] 
        @unroll for sj = 0x0:N[0x2]:(N[0x1]-0x1)
            if j+sj <= N[0x1]
                lDT1[i, j+sj] = DT[0x1][i, j+sj]
            end
        end

        @unroll for si = 0x0:N[0x1]:(N[0x2]-0x1)
            if i+si <= N[0x2]
                lDT2[i+si, j] = DT[0x2][i+si, j]
            end
        end

        lHˣ[i, j, kl] = q[i, j, k, 0x1, c]
        lHʸ[i, j, kl] = q[i, j, k, 0x2, c]
        lHᶻ[i, j, kl] = q[i, j, k, 0x3, c]
        lEˣ[i, j, kl] = q[i, j, k, 0x4, c]
        lEʸ[i, j, kl] = q[i, j, k, 0x5, c]
        lEᶻ[i, j, kl] = q[i, j, k, 0x6, c]
    end

    sync_threads()

    @inbounds if k <= G[0x3]
        dHˣijkc_update = -zero(eltype(dq))
        dHʸijkc_update = -zero(eltype(dq))
        dHᶻijkc_update = -zero(eltype(dq))
        dEˣijkc_update = -zero(eltype(dq))
        dEʸijkc_update = -zero(eltype(dq))
        dEᶻijkc_update = -zero(eltype(dq))

        invwJijkc = invwJ[i, j, k, 0x1, c]
        wJijkc = wJ[i, j, k, 0x1, c]

        wJdRdXijkc_1 = wJijkc * dRdX[i, j, k, 0x1, c]
        wJdRdXijkc_4 = wJijkc * dRdX[i, j, k, 0x4, c]
        wJdRdXijkc_7 = wJijkc * dRdX[i, j, k, 0x7, c]


        @unroll for l = 0x1:N[0x1]
            lDT1il = lDT1[i, l]
            dHˣijkc_update -= wJdRdXijkc_4 * lDT1il * lEᶻ[l, j, kl]
            dHˣijkc_update += wJdRdXijkc_7 * lDT1il * lEʸ[l, j, kl]

            dHʸijkc_update -= wJdRdXijkc_7 * lDT1il * lEˣ[l, j, kl]
            dHʸijkc_update += wJdRdXijkc_1 * lDT1il * lEᶻ[l, j, kl]

            dHᶻijkc_update -= wJdRdXijkc_1 * lDT1il * lEʸ[l, j, kl]
            dHᶻijkc_update += wJdRdXijkc_4 * lDT1il * lEˣ[l, j, kl]

            dEˣijkc_update += wJdRdXijkc_4 * lDT1il * lHᶻ[l, j, kl]
            dEˣijkc_update -= wJdRdXijkc_7 * lDT1il * lHʸ[l, j, kl]

            dEʸijkc_update += wJdRdXijkc_7 * lDT1il * lHˣ[l, j, kl]
            dEʸijkc_update -= wJdRdXijkc_1 * lDT1il * lHᶻ[l, j, kl]

            dEᶻijkc_update += wJdRdXijkc_1 * lDT1il * lHʸ[l, j, kl]
            dEᶻijkc_update -= wJdRdXijkc_4 * lDT1il * lHˣ[l, j, kl]
        end

        wJdRdXijkc_2 = wJijkc * dRdX[i, j, k, 0x2, c]
        wJdRdXijkc_5 = wJijkc * dRdX[i, j, k, 0x5, c]
        wJdRdXijkc_8 = wJijkc * dRdX[i, j, k, 0x8, c]

        @unroll for n = 0x1:N[0x2]
            lDT2jn = lDT2[j,n]
            dHˣijkc_update -= wJdRdXijkc_5 * lDT2jn * lEᶻ[i, n, kl]
            dHˣijkc_update += wJdRdXijkc_8 * lDT2jn * lEʸ[i, n, kl]

            dHʸijkc_update -= wJdRdXijkc_8 * lDT2jn * lEˣ[i, n, kl]
            dHʸijkc_update += wJdRdXijkc_2 * lDT2jn * lEᶻ[i, n, kl]

            dHᶻijkc_update -= wJdRdXijkc_2 * lDT2jn * lEʸ[i, n, kl]
            dHᶻijkc_update += wJdRdXijkc_5 * lDT2jn * lEˣ[i, n, kl]

            dEˣijkc_update += wJdRdXijkc_5 * lDT2jn * lHᶻ[i, n, kl]
            dEˣijkc_update -= wJdRdXijkc_8 * lDT2jn * lHʸ[i, n, kl]

            dEʸijkc_update += wJdRdXijkc_8 * lDT2jn * lHˣ[i, n, kl]
            dEʸijkc_update -= wJdRdXijkc_2 * lDT2jn * lHᶻ[i, n, kl]

            dEᶻijkc_update += wJdRdXijkc_2 * lDT2jn * lHʸ[i, n, kl]
            dEᶻijkc_update -= wJdRdXijkc_5 * lDT2jn * lHˣ[i, n, kl]
        end

        dq[i, j, k, 0x1, c] += invwJijkc * dHˣijkc_update
        dq[i, j, k, 0x2, c] += invwJijkc * dHʸijkc_update
        dq[i, j, k, 0x3, c] += invwJijkc * dHᶻijkc_update
        dq[i, j, k, 0x4, c] += invwJijkc * dEˣijkc_update
        dq[i, j, k, 0x5, c] += invwJijkc * dEʸijkc_update
        dq[i, j, k, 0x6, c] += invwJijkc * dEᶻijkc_update
    end
    return nothing
end

function rhs!(dq, q, grid, invwJ, DT, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)
    S = size(cell)

    start!(q, cm)

    KSTRIDE = max(256 ÷ (S[1]*S[2]), 1)
    threads = (S[1], S[2], KSTRIDE)
    blocks = (size(dq, 4), cld(S[3], KSTRIDE))
    @cuda threads=threads blocks=blocks rhs_volume_horizontal_kernel!(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(dq.data)),
        Val(size(cell)),
        Val(KSTRIDE)
    )

    ISTRIDE = max(512 ÷ (S[2]*S[3]), 1)
    threads = (ISTRIDE, S[2], S[3])
    blocks = (size(dq,4), cld(S[1], ISTRIDE))
    @cuda threads=threads blocks=blocks rhs_volume_vertical_kernel!(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(dq.data)),
        Val(size(cell)),
        Val(ISTRIDE)
    )

    finish!(q, cm)

    J = maximum([S[1]*S[2], S[1]*S[3], S[2]*S[3]])
    C = max(128 ÷ J, 1)
    @cuda threads=(J, C) blocks=cld(last(size(dq)),C) rhs_surface_kernel!( 
        parent(dq),
        parent(viewwithghosts(q)),
        fm.vmapM,
        fm.vmapP,
        boundarycodes(grid),
        parent(n),
        parent(wsJ),
        parent(invwJ),
        Val(size(cell)),
        Val(C)
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

    timeend = 0.05

    #jl # crude dt estimate
    cfl = 1 // 20
    dx = FT(2.0^(1-L)/K) 
    dt = cfl * dx / (maximum(N))^2
    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps


    #=
    RKA = (
        FT(0),
        FT(-0.7188012108672410),
        FT(-0.7785331173421570),
        FT(-0.0053282796654044),
        FT(-0.8552979934029281),
        FT(-3.9564138245774565),
        FT(-1.5780575380587385),
        FT(-2.0837094552574054),
        FT(-0.7483334182761610),
        FT(-0.7032861106563359),
        FT(0.0013917096117681),
        FT(-0.0932075369637460),
        FT(-0.9514200470875948),
        FT(-7.1151571693922548)
    )

    RKB = (
        FT(0.0367762454319673),
        FT(0.3136296607553959),
        FT(0.1531848691869027),
        FT(0.0030097086818182),
        FT(0.3326293790646110),
        FT(0.2440251405350864),
        FT(0.3718879239592277),
        FT(0.6204126221582444),
        FT(0.1524043173028741),
        FT(0.0760894927419266),
        FT(0.0077604214040978),
        FT(0.0024647284755382),
        FT(0.0780348340049386),
        FT(5.5059777270269628)
    )
    =#

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
            vtkfile["Hˣ"] = Adapt.adapt(Array, P * getindex.(q,1))
            vtkfile["Hʸ"] = Adapt.adapt(Array, P * getindex.(q,2))
            vtkfile["Hᶻ"] = Adapt.adapt(Array, P * getindex.(q,3))
            vtkfile["Eˣ"] = Adapt.adapt(Array, P * getindex.(q,4))
            vtkfile["Eʸ"] = Adapt.adapt(Array, P * getindex.(q,5))
            vtkfile["Eᶻ"] = Adapt.adapt(Array, P * getindex.(q,6))
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

    if outputvtk do_output(step, time, q) end

    for step  in ProgressBar(1:numberofsteps)
        if time + dt > timeend
            dt = timeend - time
        end
    
        for (i, stage) in enumerate(eachindex(RKA, RKB))
            @. dq *= RKA[stage]
            rhs!(dq, q, grid, invwJ, DT, cm)
            @. q += RKB[stage] * dt * dq
        end
        time += dt

        if outputvtk do_output(step, time, q) end
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

    # compute error
    _, _, wJ = components(first(volumemetrics(grid)))
    qexact = solution.(points(grid), timeend)

    #jl # TODO add sum to GridArray so the following reduction is on the device
    errf1 = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* getindex.(q .- qexact, 1).^2)), +, comm))
    errf2 = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* getindex.(q .- qexact, 2).^2)), +, comm))
    errf3 = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* getindex.(q .- qexact, 3).^2)), +, comm))
    errf4 = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* getindex.(q .- qexact, 4).^2)), +, comm))
    errf5 = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* getindex.(q .- qexact, 5).^2)), +, comm))
    errf6 = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* getindex.(q .- qexact, 6).^2)), +, comm))

    errf = maximum([errf1, errf2, errf3, errf4, errf5, errf6])
    return errf
end

let
    FT = Float64
    @assert CUDA.functional() && CUDA.has_cuda_gpu() "NVidia GPU not available"
    AT = CuArray

    N = (7, 7, 7)
    K = 4
    L = 3

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
            array type       = $AT
            outputvtx        = $outputvtk
            convergetest     = $convergetest
        """
    end

    vtkdir = "vtk_semdg_maxwells_3d$(K)x$(K)x$(K)_L$(L)"

    run(initialcondition, FT, AT, N, K, L; outputvtk = outputvtk, vtkdir, comm)

    totalcells = (K * 2^L, K * 2^L, K * 2^L)
    dofs = prod(totalcells)*prod(1 .+ N)
    err = run(initialcondition, FT, AT, N, K, L; outputvtk=false, comm)

    if rank == 0
        @info @sprintf(
            "Level %d, cells = (%2d, %2d, %2d), dof = %d, error = %.16e",
            L,
            totalcells...,
            dofs,
            err
        )
    end

    #jl # run convergence study
    if convergetest
        numlevels = 4
        N = (4, 4, 4)
        rank == 0 && @info "Starting convergence study h-refinement"
        err = zeros(FT, numlevels)
        for l = 1:numlevels
            K = 4
            L = l - 1
            totalcells = (K * 2^L, K * 2^L, K * 2^L)
            dofs = prod(totalcells)*prod(1 .+ N)
            err[l] = run(initialcondition, FT, AT, N, K, L; outputvtk=false, comm)

            if rank == 0
                @info @sprintf(
                    "Level %d, cells = (%2d, %2d, %2d), dof = %d, error = %.16e",
                    l,
                    totalcells...,
                    dofs,
                    err[l]
                )
            end
        end
        rates = log2.(err[1:numlevels-1] ./ err[2:numlevels])
        if rank == 0 && numlevels > 1
            @info "Convergence rates:\n" * join(
                ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(numlevels-1)],
                "\n",
            )
        end
    end
end
