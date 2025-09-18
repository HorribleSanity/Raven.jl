#--------------------------------Markdown Language Header-----------------------
# # 3D Maxwells equation
#--------------------------------Markdown Language Header-----------------------
using WriteVTK: num_cells_structured
using Base: sign_mask
using Adapt
using MPI
using LinearAlgebra
using Printf
using CUDA
using Raven
using KernelAbstractions
using Raven.KernelAbstractions.Extras: @unroll
using StaticArrays
using WriteVTK
using Pkg

using ProgressBars

const outputvtk = false
const bigrun = true
const convergetest = false

struct TypeWrap{T} end
TypeWrap(T) = TypeWrap{T}()
Base.:*(x::Number, ::TypeWrap{T}) where T = T(x)

const u8 = TypeWrap(UInt8)
const u16 = TypeWrap(UInt16)
const u32 = TypeWrap(UInt32)
const u64 = TypeWrap(UInt64)

const i8 = TypeWrap(Int8)
const i16 = TypeWrap(Int16)
const i32 = TypeWrap(Int32)
const i64 = TypeWrap(Int64)


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

@kernel inbounds=true unsafe_indices=true function rhs_surface!(
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
    ::Val{S},
) where {N, C, S}
    ijk, cl = @index(Local, NTuple)
    _, wg_idx = @index(Group, NTuple)
    c = (wg_idx-1u8)*C + cl

    Hxflux = @localmem eltype(dq) (N..., C)
    Hyflux = @localmem eltype(dq) (N..., C)
    Hzflux = @localmem eltype(dq) (N..., C)

    Exflux = @localmem eltype(dq) (N..., C)
    Eyflux = @localmem eltype(dq) (N..., C)
    Ezflux = @localmem eltype(dq) (N..., C)

    if ijk <= N[1u8] * N[2u8] && c <= S
        ij = ijk
        j, i = fldmod1(ij, N[1u8])

        for k = 1:N[3u8]
            z = zero(eltype(Hxflux))
            Hxflux[i, j, k, cl] = z
            Hyflux[i, j, k, cl] = z
            Hzflux[i, j, k, cl] = z

            Exflux[i, j, k, cl] = z
            Eyflux[i, j, k, cl] = z
            Ezflux[i, j, k, cl] = z
        end
    end
    @synchronize

    c = (wg_idx-1u8)*C + cl
    if ijk <= N[2u8]*N[3u8] && c <= S
        alpha = 1.0
        # face with r = -1
        jk = ijk
        i = 1
        k, j = fldmod1(jk, N[2u8])

        fid = jk

        idP = vmapP[fid, c]
        idB = mapB[1u8, c]

        if idB == 1u8
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4u8, c]
            dEy = -2 * q[i, j, k, 5u8, c]
            dEz = -2 * q[i, j, k, 6u8, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1u8]*N[2u8]*N[3u8])
            Pk, Pij  = fldmod1(Pijk, N[1u8]*N[2u8])
            Pj, Pi   = fldmod1(Pij,  N[1u8])

            dHx = q[Pi, Pj, Pk, 1u8, Pc] - q[i, j, k, 1u8, c]
            dHy = q[Pi, Pj, Pk, 2u8, Pc] - q[i, j, k, 2u8, c]
            dHz = q[Pi, Pj, Pk, 3u8, Pc] - q[i, j, k, 3u8, c]

            dEx = q[Pi, Pj, Pk, 4u8, Pc] - q[i, j, k, 4u8, c]
            dEy = q[Pi, Pj, Pk, 5u8, Pc] - q[i, j, k, 5u8, c]
            dEz = q[Pi, Pj, Pk, 6u8, Pc] - q[i, j, k, 6u8, c]
        end

        nx = n[1u8, fid, c]
        ny = n[2u8, fid, c]
        nz = n[3u8, fid, c]
        wsJf = wsJ[1u8, fid, c]

        invwJijkc = invwJ[i, j, k, 1u8, c]
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
        i = N[1u8]
        fid = N[2u8]*N[3u8] + jk

        idP = vmapP[fid, c]
        idB = mapB[2u8, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4u8, c]
            dEy = -2 * q[i, j, k, 5u8, c]
            dEz = -2 * q[i, j, k, 6u8, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1u8]*N[2u8]*N[3u8])
            Pk, Pij  = fldmod1(Pijk, N[1u8]*N[2u8])
            Pj, Pi   = fldmod1(Pij,  N[1u8])

            dHx = q[Pi, Pj, Pk, 1u8, Pc] - q[i, j, k, 1u8, c]
            dHy = q[Pi, Pj, Pk, 2u8, Pc] - q[i, j, k, 2u8, c]
            dHz = q[Pi, Pj, Pk, 3u8, Pc] - q[i, j, k, 3u8, c]

            dEx = q[Pi, Pj, Pk, 4u8, Pc] - q[i, j, k, 4u8, c]
            dEy = q[Pi, Pj, Pk, 5u8, Pc] - q[i, j, k, 5u8, c]
            dEz = q[Pi, Pj, Pk, 6u8, Pc] - q[i, j, k, 6u8, c]

        end

        nx = n[1u8, fid, c]
        ny = n[2u8, fid, c]
        nz = n[3u8, fid, c]
        wsJf = wsJ[1u8, fid, c]

        invwJijkc = invwJ[i, j, k, 1u8, c]
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

    @synchronize

    c = (wg_idx-1u8)*C + cl
    if ijk <= N[1u8]*N[3u8] && c <= S
        alpha = 1.0
        # face with s = -1
        ik = ijk
        j = 1
        k, i = fldmod1(ik, N[1u8])

        fid = 2 * N[2u8] * N[3u8] + ik

        idP = vmapP[fid, c]
        idB = mapB[3u8, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4u8, c]
            dEy = -2 * q[i, j, k, 5u8, c]
            dEz = -2 * q[i, j, k, 6u8, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1u8]*N[2u8]*N[3u8])
            Pk, Pij  = fldmod1(Pijk, N[1u8]*N[2u8])
            Pj, Pi   = fldmod1(Pij,  N[1u8])

            dHx = q[Pi, Pj, Pk, 1u8, Pc] - q[i, j, k, 1u8, c]
            dHy = q[Pi, Pj, Pk, 2u8, Pc] - q[i, j, k, 2u8, c]
            dHz = q[Pi, Pj, Pk, 3u8, Pc] - q[i, j, k, 3u8, c]

            dEx = q[Pi, Pj, Pk, 4u8, Pc] - q[i, j, k, 4u8, c]
            dEy = q[Pi, Pj, Pk, 5u8, Pc] - q[i, j, k, 5u8, c]
            dEz = q[Pi, Pj, Pk, 6u8, Pc] - q[i, j, k, 6u8, c]

        end

        nx = n[1u8, fid, c]
        ny = n[2u8, fid, c]
        nz = n[3u8, fid, c]
        wsJf = wsJ[1u8, fid, c]

        invwJijkc = invwJ[i, j, k, 1u8, c]
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
        j = N[2u8]
        fid = 2*N[2u8]*N[3u8] + N[1u8]*N[3u8]  + ik

        idP = vmapP[fid, c]
        idB = mapB[4u8, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4u8, c]
            dEy = -2 * q[i, j, k, 5u8, c]
            dEz = -2 * q[i, j, k, 6u8, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1u8]*N[2u8]*N[3u8])
            Pk, Pij  = fldmod1(Pijk, N[1u8]*N[2u8])
            Pj, Pi   = fldmod1(Pij,  N[1u8])

            dHx = q[Pi, Pj, Pk, 1u8, Pc] - q[i, j, k, 1u8, c]
            dHy = q[Pi, Pj, Pk, 2u8, Pc] - q[i, j, k, 2u8, c]
            dHz = q[Pi, Pj, Pk, 3u8, Pc] - q[i, j, k, 3u8, c]

            dEx = q[Pi, Pj, Pk, 4u8, Pc] - q[i, j, k, 4u8, c]
            dEy = q[Pi, Pj, Pk, 5u8, Pc] - q[i, j, k, 5u8, c]
            dEz = q[Pi, Pj, Pk, 6u8, Pc] - q[i, j, k, 6u8, c]

        end

        nx = n[1u8, fid, c]
        ny = n[2u8, fid, c]
        nz = n[3u8, fid, c]
        wsJf = wsJ[1u8, fid, c]

        invwJijkc = invwJ[i, j, k, 1u8, c]
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

    @synchronize

    c = (wg_idx-1u8)*C + cl
    if ijk <= N[1u8]*N[2u8] && c <= S
        alpha = 1.0
        # face with t = -1
        ij = ijk

        j, i = fldmod1(ij, N[1u8])
        k = 1

        fid = 2 * (N[2u8] * N[3u8] + N[1u8] * N[3u8]) + ij

        idP = vmapP[fid, c]
        idB = mapB[5u8, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4u8, c]
            dEy = -2 * q[i, j, k, 5u8, c]
            dEz = -2 * q[i, j, k, 6u8, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1u8]*N[2u8]*N[3u8])
            Pk, Pij  = fldmod1(Pijk, N[1u8]*N[2u8])
            Pj, Pi   = fldmod1(Pij,  N[1u8])

            dHx = q[Pi, Pj, Pk, 1u8, Pc] - q[i, j, k, 1u8, c]
            dHy = q[Pi, Pj, Pk, 2u8, Pc] - q[i, j, k, 2u8, c]
            dHz = q[Pi, Pj, Pk, 3u8, Pc] - q[i, j, k, 3u8, c]

            dEx = q[Pi, Pj, Pk, 4u8, Pc] - q[i, j, k, 4u8, c]
            dEy = q[Pi, Pj, Pk, 5u8, Pc] - q[i, j, k, 5u8, c]
            dEz = q[Pi, Pj, Pk, 6u8, Pc] - q[i, j, k, 6u8, c]
        end

        nx = n[1u8, fid, c]
        ny = n[2u8, fid, c]
        nz = n[3u8, fid, c]
        wsJf = wsJ[1u8, fid, c]

        invwJijkc = invwJ[i, j, k, 1u8, c]
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
        k = N[3u8]
        fid = 2 * (N[2u8] * N[3u8] + N[1u8] * N[3u8]) + N[1u8] * N[2u8]  + ij

        idP = vmapP[fid, c]
        idB = mapB[6u8, c]

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4u8, c]
            dEy = -2 * q[i, j, k, 5u8, c]
            dEz = -2 * q[i, j, k, 6u8, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1u8]*N[2u8]*N[3u8])
            Pk, Pij  = fldmod1(Pijk, N[1u8]*N[2u8])
            Pj, Pi   = fldmod1(Pij,  N[1u8])

            dHx = q[Pi, Pj, Pk, 1u8, Pc] - q[i, j, k, 1u8, c]
            dHy = q[Pi, Pj, Pk, 2u8, Pc] - q[i, j, k, 2u8, c]
            dHz = q[Pi, Pj, Pk, 3u8, Pc] - q[i, j, k, 3u8, c]

            dEx = q[Pi, Pj, Pk, 4u8, Pc] - q[i, j, k, 4u8, c]
            dEy = q[Pi, Pj, Pk, 5u8, Pc] - q[i, j, k, 5u8, c]
            dEz = q[Pi, Pj, Pk, 6u8, Pc] - q[i, j, k, 6u8, c]
        end

        nx = n[1u8, fid, c]
        ny = n[2u8, fid, c]
        nz = n[3u8, fid, c]
        wsJf = wsJ[1u8, fid, c]

        invwJijkc = invwJ[i, j, k, 1u8, c]
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

    @synchronize

    c = (wg_idx-1u8)*C + cl
    ij = ijk
    if ij <= N[1u8] * N[2u8] && c <= S
        j, i = fldmod1(ij, N[1u8])

        for k = 1:N[3u8]
            dq[i, j, k, 1u8, c] += Hxflux[i, j, k, cl]
            dq[i, j, k, 2u8, c] += Hyflux[i, j, k, cl]
            dq[i, j, k, 3u8, c] += Hzflux[i, j, k, cl]
            dq[i, j, k, 4u8, c] += Exflux[i, j, k, cl]
            dq[i, j, k, 5u8, c] += Eyflux[i, j, k, cl]
            dq[i, j, k, 6u8, c] += Ezflux[i, j, k, cl]
        end
    end
end

@kernel inbounds=true unsafe_indices=true  function rhs_volume_vertical!(
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
    il, j, k = @index(Local, NTuple)
    iblockidx, c, _ = @index(Group, NTuple)
    i = (iblockidx-1u8)*ISTRIDE + il

    il = Int32(il)
    i = Int32(i)
    j = Int32(j)
    k = Int32(k)
    iblockidx = Int32(iblockidx)
    c = Int32(c)

    lDT3 = @localmem eltype(dq) (N[3u8], N[3u8])

    lHˣ = @localmem eltype(dq) (ISTRIDE, N[2u8], N[3u8])
    lHʸ = @localmem eltype(dq) (ISTRIDE, N[2u8], N[3u8])
    lHᶻ = @localmem eltype(dq) (ISTRIDE, N[2u8], N[3u8])

    lEˣ = @localmem eltype(dq) (ISTRIDE, N[2u8], N[3u8])
    lEʸ = @localmem eltype(dq) (ISTRIDE, N[2u8], N[3u8])
    lEᶻ = @localmem eltype(dq) (ISTRIDE, N[2u8], N[3u8])

    if i <= G[1u8]
        for sj = 0u8:N[2u8]:(N[3u8]-1u8)
            if j+sj <= N[3u8] && il == 1
                lDT3[j+sj, k] = DT[3u8][j+sj, k]
            end
        end

        lHˣ[il, j, k] = q[i, j, k, 1u8, c]
        lHʸ[il, j, k] = q[i, j, k, 2u8, c]
        lHᶻ[il, j, k] = q[i, j, k, 3u8, c]
        lEˣ[il, j, k] = q[i, j, k, 4u8, c]
        lEʸ[il, j, k] = q[i, j, k, 5u8, c]
        lEᶻ[il, j, k] = q[i, j, k, 6u8, c]
    end

    @synchronize

    i = (iblockidx-1u8)*ISTRIDE + il
     if i <= G[1u8]
        dHˣijkc_update = -zero(eltype(dq))
        dHʸijkc_update = -zero(eltype(dq))
        dHᶻijkc_update = -zero(eltype(dq))
        dEˣijkc_update = -zero(eltype(dq))
        dEʸijkc_update = -zero(eltype(dq))
        dEᶻijkc_update = -zero(eltype(dq))

        invwJijkc = invwJ[i, j, k, 1u8, c]
        wJijkc = wJ[i, j, k, 1u8, c]

        wJdRdXijkc_3 = wJijkc * dRdX[i, j, k, 3u8, c]
        wJdRdXijkc_6 = wJijkc * dRdX[i, j, k, 6u8, c]
        wJdRdXijkc_9 = wJijkc * dRdX[i, j, k, 9u8, c]

        @unroll for m = 1u8:N[3u8]
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

        dq[i, j, k, 1u8, c] += invwJijkc * dHˣijkc_update
        dq[i, j, k, 2u8, c] += invwJijkc * dHʸijkc_update
        dq[i, j, k, 3u8, c] += invwJijkc * dHᶻijkc_update
        dq[i, j, k, 4u8, c] += invwJijkc * dEˣijkc_update
        dq[i, j, k, 5u8, c] += invwJijkc * dEʸijkc_update
        dq[i, j, k, 6u8, c] += invwJijkc * dEᶻijkc_update
    end
end

@kernel inbounds=true unsafe_indices=true  function rhs_volume_horizontal!(
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
    i, j, kl = @index(Local, NTuple)
    c, kblockidx = @index(Group, NTuple)
    k = (kblockidx - 1u8) * KSTRIDE + kl

    kl = Int32(kl)
    i = Int32(i)
    j = Int32(j)
    k = Int32(k)
    kblockidx = Int32(kblockidx)
    c = Int32(c)

    lDT1 = @localmem eltype(dq) (N[1u8], N[1u8])
    lDT2 = @localmem eltype(dq) (N[2u8], N[2u8])

    lHˣ = @localmem eltype(dq) (N[1u8], N[2u8], KSTRIDE)
    lHʸ = @localmem eltype(dq) (N[1u8], N[2u8], KSTRIDE)
    lHᶻ = @localmem eltype(dq) (N[1u8], N[2u8], KSTRIDE)

    lEˣ = @localmem eltype(dq) (N[1u8], N[2u8], KSTRIDE)
    lEʸ = @localmem eltype(dq) (N[1u8], N[2u8], KSTRIDE)
    lEᶻ = @localmem eltype(dq) (N[1u8], N[2u8], KSTRIDE)

    if k <= G[3u8]
        @unroll for sj = 0u8:N[2u8]:(N[1u8]-1u8)
            if j+sj <= N[1u8]
                lDT1[i, j+sj] = DT[1u8][i, j+sj]
            end
        end

        @unroll for si = 0u8:N[1u8]:(N[2u8]-1u8)
            if i+si <= N[2u8]
                lDT2[i+si, j] = DT[2u8][i+si, j]
            end
        end

        lHˣ[i, j, kl] = q[i, j, k, 1u8, c]
        lHʸ[i, j, kl] = q[i, j, k, 2u8, c]
        lHᶻ[i, j, kl] = q[i, j, k, 3u8, c]
        lEˣ[i, j, kl] = q[i, j, k, 4u8, c]
        lEʸ[i, j, kl] = q[i, j, k, 5u8, c]
        lEᶻ[i, j, kl] = q[i, j, k, 6u8, c]
    end

    @synchronize

    k = (kblockidx - 1u8) * KSTRIDE + kl
    if k <= G[3u8]
        dHˣijkc_update = -zero(eltype(dq))
        dHʸijkc_update = -zero(eltype(dq))
        dHᶻijkc_update = -zero(eltype(dq))
        dEˣijkc_update = -zero(eltype(dq))
        dEʸijkc_update = -zero(eltype(dq))
        dEᶻijkc_update = -zero(eltype(dq))

        invwJijkc = invwJ[i, j, k, 1u8, c]
        wJijkc = wJ[i, j, k, 1u8, c]

        wJdRdXijkc_1 = wJijkc * dRdX[i, j, k, 1u8, c]
        wJdRdXijkc_4 = wJijkc * dRdX[i, j, k, 4u8, c]
        wJdRdXijkc_7 = wJijkc * dRdX[i, j, k, 7u8, c]


        @unroll for l = 1u8:N[1u8]
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

        wJdRdXijkc_2 = wJijkc * dRdX[i, j, k, 2u8, c]
        wJdRdXijkc_5 = wJijkc * dRdX[i, j, k, 5u8, c]
        wJdRdXijkc_8 = wJijkc * dRdX[i, j, k, 8u8, c]

        @unroll for n = 1u8:N[2u8]
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

        dq[i, j, k, 1u8, c] += invwJijkc * dHˣijkc_update
        dq[i, j, k, 2u8, c] += invwJijkc * dHʸijkc_update
        dq[i, j, k, 3u8, c] += invwJijkc * dHᶻijkc_update
        dq[i, j, k, 4u8, c] += invwJijkc * dEˣijkc_update
        dq[i, j, k, 5u8, c] += invwJijkc * dEʸijkc_update
        dq[i, j, k, 6u8, c] += invwJijkc * dEᶻijkc_update
    end
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
    workgroup = (S[1], S[2], KSTRIDE)
    blocks = (size(dq, 4), cld(S[3], KSTRIDE),1)
    rhs_volume_horizontal!(backend, workgroup)(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(dq.data)),
        Val(size(cell)),
        Val(KSTRIDE);
        ndrange = workgroup .* blocks
    )

    ISTRIDE = max(256 ÷ (S[2]*S[3]), 1)
    workgroup = (ISTRIDE, S[2], S[3])
    blocks = (cld(S[1], ISTRIDE), size(dq,4), 1)
    rhs_volume_vertical!(backend, workgroup)(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(dq.data)),
        Val(size(cell)),
        Val(ISTRIDE);
        ndrange = workgroup .* blocks
    )

    finish!(q, cm)

    J = maximum([S[1]*S[2], S[1]*S[3], S[2]*S[3]])
    C = max(128 ÷ J, 1)
    workgroup = (J, C)
    blocks = (1, cld(last(size(dq)),C))
    rhs_surface!(backend, workgroup)(
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
        Val(last(size(dq)));
        ndrange = workgroup .* blocks
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
    cfl = 1 // 2
    dx = FT(2.0^(1-L)/K)
    dt = cfl * dx / (maximum(N))^2
    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps


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

    #=
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
    =#

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

    if !bigrun
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
end

let
    FT = Float64
    AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
        CUDA.allowscalar(false)
    end

    if bigrun || outputvtk
        N = (7, 7, 7)
        K = 4
        L = 3

        vtkdir = "vtk_semdg_maxwells_3d$(K)x$(K)x$(K)_L$(L)"

        totalcells = (K * 2^L, K * 2^L, K * 2^L)
        dofs = prod(totalcells)*prod(1 .+ N)*6
        if rank == 0
            @info @sprintf(
                "Level %d, cells = (%2d, %2d, %2d), dof = %1.2e, size of q = %1.2f GB",
                L,
                totalcells...,
                dofs,
                dofs*sizeof(FT)*1e-9,
            )

            @info """Configuration:
                precision        = $FT
                array type       = $AT
                outputvtx        = $outputvtk
                convergetest     = $convergetest
            """
        end
        run(initialcondition, FT, AT, N, K, L; outputvtk = outputvtk, vtkdir, comm)
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
