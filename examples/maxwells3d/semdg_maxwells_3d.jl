using WriteVTK: num_cells_structured
using CUDA: initialize_context
#--------------------------------Markdown Language Header-----------------------
# # 3D Maxwells equation
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

const outputvtk = false 
const convergetest = true 


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
    c = (blockIdx().x-1)*blockDim().y + cl

    Hxflux = CuStaticSharedArray(eltype(dq), (N..., C)) # @localmem eltype(dq) (N..., C)
    Hyflux = CuStaticSharedArray(eltype(dq), (N..., C)) #  @localmem eltype(dq) (N..., C)
    Hzflux = CuStaticSharedArray(eltype(dq), (N..., C)) #  @localmem eltype(dq) (N..., C)

    Exflux = CuStaticSharedArray(eltype(dq), (N..., C)) #  @localmem eltype(dq) (N..., C)
    Eyflux = CuStaticSharedArray(eltype(dq), (N..., C)) #  @localmem eltype(dq) (N..., C)
    Ezflux = CuStaticSharedArray(eltype(dq), (N..., C)) #  @localmem eltype(dq) (N..., C)

    @inbounds if ijk <= N[1] * N[2]
        ij = ijk
        j, i = fldmod1(ij, N[1])

        for k = 1:N[3]
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

    @inbounds if ijk <= N[2]*N[3]
        alpha = 1.0
        # face with r = -1
        jk = ijk
        i = 1
        k, j = fldmod1(jk, N[2])

        fid = jk 

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[1, c]     # global GPU memory access

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4, c]
            dEy = -2 * q[i, j, k, 5, c]
            dEz = -2 * q[i, j, k, 6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1]*N[2]*N[3])
            Pk, Pij  = fldmod1(Pijk, N[1]*N[2])
            Pj, Pi   = fldmod1(Pij,  N[1])

            dHx = q[Pi, Pj, Pk, 1, Pc] - q[i, j, k, 1, c]
            dHy = q[Pi, Pj, Pk, 2, Pc] - q[i, j, k, 2, c]
            dHz = q[Pi, Pj, Pk, 3, Pc] - q[i, j, k, 3, c]

            dEx = q[Pi, Pj, Pk, 4, Pc] - q[i, j, k, 4, c]
            dEy = q[Pi, Pj, Pk, 5, Pc] - q[i, j, k, 5, c]
            dEz = q[Pi, Pj, Pk, 6, Pc] - q[i, j, k, 6, c]
        end

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]
        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
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
        i = N[1]
        fid = N[2]*N[3] + jk

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[2, c]     # global GPU memory access

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4, c]
            dEy = -2 * q[i, j, k, 5, c]
            dEz = -2 * q[i, j, k, 6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1]*N[2]*N[3])
            Pk, Pij  = fldmod1(Pijk, N[1]*N[2])
            Pj, Pi   = fldmod1(Pij,  N[1])

            dHx = q[Pi, Pj, Pk, 1, Pc] - q[i, j, k, 1, c]
            dHy = q[Pi, Pj, Pk, 2, Pc] - q[i, j, k, 2, c]
            dHz = q[Pi, Pj, Pk, 3, Pc] - q[i, j, k, 3, c]

            dEx = q[Pi, Pj, Pk, 4, Pc] - q[i, j, k, 4, c]
            dEy = q[Pi, Pj, Pk, 5, Pc] - q[i, j, k, 5, c]
            dEz = q[Pi, Pj, Pk, 6, Pc] - q[i, j, k, 6, c]

        end

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]
        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
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

    @inbounds if ijk <= N[1]*N[3]
        alpha = 1.0
        # face with s = -1
        ik = ijk
        j = 1
        k, i = fldmod1(ik, N[1])

        fid = 2 * N[2] * N[3] + ik

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[3, c]     # global GPU memory access

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4, c]
            dEy = -2 * q[i, j, k, 5, c]
            dEz = -2 * q[i, j, k, 6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1]*N[2]*N[3])
            Pk, Pij  = fldmod1(Pijk, N[1]*N[2])
            Pj, Pi   = fldmod1(Pij,  N[1])

            dHx = q[Pi, Pj, Pk, 1, Pc] - q[i, j, k, 1, c]
            dHy = q[Pi, Pj, Pk, 2, Pc] - q[i, j, k, 2, c]
            dHz = q[Pi, Pj, Pk, 3, Pc] - q[i, j, k, 3, c]

            dEx = q[Pi, Pj, Pk, 4, Pc] - q[i, j, k, 4, c]
            dEy = q[Pi, Pj, Pk, 5, Pc] - q[i, j, k, 5, c]
            dEz = q[Pi, Pj, Pk, 6, Pc] - q[i, j, k, 6, c]

        end

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]
        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
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
        j = N[2]
        fid = 2*N[2]*N[3] + N[1]*N[3]  + ik

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[4, c]     # global GPU memory access

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4, c]
            dEy = -2 * q[i, j, k, 5, c]
            dEz = -2 * q[i, j, k, 6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1]*N[2]*N[3])
            Pk, Pij  = fldmod1(Pijk, N[1]*N[2])
            Pj, Pi   = fldmod1(Pij,  N[1])

            dHx = q[Pi, Pj, Pk, 1, Pc] - q[i, j, k, 1, c]
            dHy = q[Pi, Pj, Pk, 2, Pc] - q[i, j, k, 2, c]
            dHz = q[Pi, Pj, Pk, 3, Pc] - q[i, j, k, 3, c]

            dEx = q[Pi, Pj, Pk, 4, Pc] - q[i, j, k, 4, c]
            dEy = q[Pi, Pj, Pk, 5, Pc] - q[i, j, k, 5, c]
            dEz = q[Pi, Pj, Pk, 6, Pc] - q[i, j, k, 6, c]

        end

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]
        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
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

    @inbounds if ijk <= N[1]*N[2]
        alpha = 1.0
        # face with t = -1
        ij = ijk

        j, i = fldmod1(ij, N[1])
        k = 1

        fid = 2 * (N[2] * N[3] + N[1] * N[3]) + ij 

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[5, c]     # global GPU memory access

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4, c]
            dEy = -2 * q[i, j, k, 5, c]
            dEz = -2 * q[i, j, k, 6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1]*N[2]*N[3])
            Pk, Pij  = fldmod1(Pijk, N[1]*N[2])
            Pj, Pi   = fldmod1(Pij,  N[1])

            dHx = q[Pi, Pj, Pk, 1, Pc] - q[i, j, k, 1, c]
            dHy = q[Pi, Pj, Pk, 2, Pc] - q[i, j, k, 2, c]
            dHz = q[Pi, Pj, Pk, 3, Pc] - q[i, j, k, 3, c]

            dEx = q[Pi, Pj, Pk, 4, Pc] - q[i, j, k, 4, c]
            dEy = q[Pi, Pj, Pk, 5, Pc] - q[i, j, k, 5, c]
            dEz = q[Pi, Pj, Pk, 6, Pc] - q[i, j, k, 6, c]
        end

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]
        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
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
        k = N[3]
        fid = 2 * (N[2] * N[3] + N[1] * N[3]) + N[1] * N[2]  + ij

        idP = vmapP[fid, c]  # global GPU memory access
        idB = mapB[6, c]     # global GPU memory access

        if idB == 1
            dHx = zero(eltype(Hxflux))
            dHy = zero(eltype(Hyflux))
            dHz = zero(eltype(Hzflux))

            dEx = -2 * q[i, j, k, 4, c]
            dEy = -2 * q[i, j, k, 5, c]
            dEz = -2 * q[i, j, k, 6, c]
        else
            Pc, Pijk = fldmod1(idP,  N[1]*N[2]*N[3])
            Pk, Pij  = fldmod1(Pijk, N[1]*N[2])
            Pj, Pi   = fldmod1(Pij,  N[1])

            dHx = q[Pi, Pj, Pk, 1, Pc] - q[i, j, k, 1, c]
            dHy = q[Pi, Pj, Pk, 2, Pc] - q[i, j, k, 2, c]
            dHz = q[Pi, Pj, Pk, 3, Pc] - q[i, j, k, 3, c]

            dEx = q[Pi, Pj, Pk, 4, Pc] - q[i, j, k, 4, c]
            dEy = q[Pi, Pj, Pk, 5, Pc] - q[i, j, k, 5, c]
            dEz = q[Pi, Pj, Pk, 6, Pc] - q[i, j, k, 6, c]
        end

        nx = n[1, fid, c]
        ny = n[2, fid, c]
        nz = n[3, fid, c]
        wsJf = wsJ[1, fid, c]

        invwJijkc = invwJ[i, j, k, 1, c]
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
    @inbounds if ij <= N[1] * N[2]
        j, i = fldmod1(ij, N[1])

        for k = 1:N[3]
            dq[i, j, k, 1, c] += Hxflux[i, j, k, cl]
            dq[i, j, k, 2, c] += Hyflux[i, j, k, cl]
            dq[i, j, k, 3, c] += Hzflux[i, j, k, cl]
            dq[i, j, k, 4, c] += Exflux[i, j, k, cl]
            dq[i, j, k, 5, c] += Eyflux[i, j, k, cl]
            dq[i, j, k, 6, c] += Ezflux[i, j, k, cl]
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
    ::Val{N},
    ::Val{ISTRIDE},
) where {N, ISTRIDE}
    j, k = threadIdx() #@index(Local, NTuple)
    c, i = blockIdx() #@index(Global, NTuple)

    il = 0x1
    
    lDT3 = CuStaticSharedArray(Float64, (N[3], N[3]))

    lHˣ = CuStaticSharedArray(Float64, (ISTRIDE, N[2], N[3]))
    lHʸ = CuStaticSharedArray(Float64, (ISTRIDE, N[2], N[3]))
    lHᶻ = CuStaticSharedArray(Float64, (ISTRIDE, N[2], N[3]))

    lEˣ = CuStaticSharedArray(Float64, (ISTRIDE, N[2], N[3]))
    lEʸ = CuStaticSharedArray(Float64, (ISTRIDE, N[2], N[3]))
    lEᶻ = CuStaticSharedArray(Float64, (ISTRIDE, N[2], N[3]))

    @inbounds begin
        for sj = 0x0:N[2]:(N[3]-0x1)
            if j+sj <= N[3]
                lDT3[j+sj, k] = DT[3][j+sj, k]
            end
        end

        # loading into local mem data (i, j, k) from global index cell c
        lHˣ[il, j, k] = q[i, j, k, 0x1, c]
        lHʸ[il, j, k] = q[i, j, k, 0x2, c]
        lHᶻ[il, j, k] = q[i, j, k, 0x3, c]
        lEˣ[il, j, k] = q[i, j, k, 0x4, c]
        lEʸ[il, j, k] = q[i, j, k, 0x5, c]
        lEᶻ[il, j, k] = q[i, j, k, 0x6, c]
    end

    sync_threads()

    @inbounds begin
        dHˣijkc_update = -zero(Float64)
        dHʸijkc_update = -zero(Float64)
        dHᶻijkc_update = -zero(Float64)
        dEˣijkc_update = -zero(Float64)
        dEʸijkc_update = -zero(Float64)
        dEᶻijkc_update = -zero(Float64)

        invwJijkc = invwJ[i, j, k, 1, c]
        wJijkc = wJ[i, j, k, 1, c]

        # dHˣdt = - D_y Eᶻ + D_z Eʸ = - (r_y D_r + s_y D_s + t_y D_t) Eᶻ + (r_z D_r + s_z D_s + t_z D_t) Eʸ
        # dHʸdt = - D_z Eˣ + D_x Eᶻ = - (r_z D_r + s_z D_s + t_z D_t) Eˣ + (r_x D_r + s_x D_s + t_x D_t) Eᶻ
        # dHᶻdt = - D_x Eʸ + D_y Eˣ = - (r_x D_r + s_x D_s + t_x D_t) Eʸ + (r_y D_r + s_y D_s + t_y D_t) Eˣ


        # dEˣdt =   D_y Hᶻ - D_z Hʸ =   (r_y D_r + s_y D_s + t_y D_t) Hᶻ - (r_z D_r + s_z D_s + t_z D_t) Hʸ
        # dEʸdt =   D_z Hˣ - D_x Hᶻ =   (r_z D_r + s_z D_s + t_z D_t) Hˣ - (r_x D_r + s_x D_s + t_x D_t) Hᶻ
        # dEᶻdt =   D_x Hʸ - D_y Hˣ =   (r_x D_r + s_x D_s + t_x D_t) Hʸ - (r_y D_r + s_y D_s + t_y D_t) Hˣ

        for m = 0x1:N[3]
            dHˣijkc_update -= wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lEᶻ[il, j, m]
            dHˣijkc_update += wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lEʸ[il, j, m]

            dHʸijkc_update -= wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lEˣ[il, j, m]
            dHʸijkc_update += wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lEᶻ[il, j, m]

            dHᶻijkc_update -= wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lEʸ[il, j, m]
            dHᶻijkc_update += wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lEˣ[il, j, m]

            dEˣijkc_update += wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lHᶻ[il, j, m]
            dEˣijkc_update -= wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lHʸ[il, j, m]

            dEʸijkc_update += wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lHˣ[il, j, m]
            dEʸijkc_update -= wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lHᶻ[il, j, m]

            dEᶻijkc_update += wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lHʸ[il, j, m]
            dEᶻijkc_update -= wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lHˣ[il, j, m]
        end

        dq[i, j, k, 1, c] += invwJijkc * dHˣijkc_update
        dq[i, j, k, 2, c] += invwJijkc * dHʸijkc_update
        dq[i, j, k, 3, c] += invwJijkc * dHᶻijkc_update
        dq[i, j, k, 4, c] += invwJijkc * dEˣijkc_update
        dq[i, j, k, 5, c] += invwJijkc * dEʸijkc_update
        dq[i, j, k, 6, c] += invwJijkc * dEᶻijkc_update
    end
    return nothing
end


#TODO: add striding, gpuize time stepping, fix memory transfers from host to dev
function rhs_volume_horizontal_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    ::Val{N},
    ::Val{KSTRIDE},
) where {N, KSTRIDE}
    i, j = threadIdx() #@index(Local, NTuple)
    c, k = blockIdx() #@index(Global, NTuple)
    kl = 0x1
    #c = blockIdx().x #@index(Global, NTuple)

    lDT1 = CuStaticSharedArray(Float64, (N[1], N[1]))
    lDT2 = CuStaticSharedArray(Float64, (N[2], N[2]))

    lHˣ = CuStaticSharedArray(Float64, (N[1], N[2], KSTRIDE))
    lHʸ = CuStaticSharedArray(Float64, (N[1], N[2], KSTRIDE))
    lHᶻ = CuStaticSharedArray(Float64, (N[1], N[2], KSTRIDE))

    lEˣ = CuStaticSharedArray(Float64, (N[1], N[2], KSTRIDE))
    lEʸ = CuStaticSharedArray(Float64, (N[1], N[2], KSTRIDE))
    lEᶻ = CuStaticSharedArray(Float64, (N[1], N[2], KSTRIDE))

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1)
            if j+sj <= N[1]
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i+si <= N[2]
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        # loading into local mem data (i, j, k) from global index cell c
        lHˣ[i, j, kl] = q[i, j, k, 0x1, c]
        lHʸ[i, j, kl] = q[i, j, k, 0x2, c]
        lHᶻ[i, j, kl] = q[i, j, k, 0x3, c]
        lEˣ[i, j, kl] = q[i, j, k, 0x4, c]
        lEʸ[i, j, kl] = q[i, j, k, 0x5, c]
        lEᶻ[i, j, kl] = q[i, j, k, 0x6, c]
    end

    sync_threads()

    @inbounds begin
        dHˣijkc_update = -zero(eltype(dq))
        dHʸijkc_update = -zero(eltype(dq))
        dHᶻijkc_update = -zero(eltype(dq))
        dEˣijkc_update = -zero(eltype(dq))
        dEʸijkc_update = -zero(eltype(dq))
        dEᶻijkc_update = -zero(eltype(dq))

        invwJijkc = invwJ[i, j, k, 1, c]
        wJijkc = wJ[i, j, k, 1, c]

        # dHˣdt = - D_y Eᶻ + D_z Eʸ = - (r_y D_r + s_y D_s + t_y D_t) Eᶻ + (r_z D_r + s_z D_s + t_z D_t) Eʸ
        # dHʸdt = - D_z Eˣ + D_x Eᶻ = - (r_z D_r + s_z D_s + t_z D_t) Eˣ + (r_x D_r + s_x D_s + t_x D_t) Eᶻ
        # dHᶻdt = - D_x Eʸ + D_y Eˣ = - (r_x D_r + s_x D_s + t_x D_t) Eʸ + (r_y D_r + s_y D_s + t_y D_t) Eˣ


        # dEˣdt =   D_y Hᶻ - D_z Hʸ =   (r_y D_r + s_y D_s + t_y D_t) Hᶻ - (r_z D_r + s_z D_s + t_z D_t) Hʸ
        # dEʸdt =   D_z Hˣ - D_x Hᶻ =   (r_z D_r + s_z D_s + t_z D_t) Hˣ - (r_x D_r + s_x D_s + t_x D_t) Hᶻ
        # dEᶻdt =   D_x Hʸ - D_y Hˣ =   (r_x D_r + s_x D_s + t_x D_t) Hʸ - (r_y D_r + s_y D_s + t_y D_t) Hˣ

        for l = 0x1:N[1]
            dHˣijkc_update -= wJijkc * dRdX[i, j, k, 4, c] * lDT1[i, l] * lEᶻ[l, j, kl]
            dHˣijkc_update += wJijkc * dRdX[i, j, k, 7, c] * lDT1[i, l] * lEʸ[l, j, kl]

            dHʸijkc_update -= wJijkc * dRdX[i, j, k, 7, c] * lDT1[i, l] * lEˣ[l, j, kl]
            dHʸijkc_update += wJijkc * dRdX[i, j, k, 1, c] * lDT1[i, l] * lEᶻ[l, j, kl]

            dHᶻijkc_update -= wJijkc * dRdX[i, j, k, 1, c] * lDT1[i, l] * lEʸ[l, j, kl]
            dHᶻijkc_update += wJijkc * dRdX[i, j, k, 4, c] * lDT1[i, l] * lEˣ[l, j, kl]

            dEˣijkc_update += wJijkc * dRdX[i, j, k, 4, c] * lDT1[i, l] * lHᶻ[l, j, kl]
            dEˣijkc_update -= wJijkc * dRdX[i, j, k, 7, c] * lDT1[i, l] * lHʸ[l, j, kl]

            dEʸijkc_update += wJijkc * dRdX[i, j, k, 7, c] * lDT1[i, l] * lHˣ[l, j, kl]
            dEʸijkc_update -= wJijkc * dRdX[i, j, k, 1, c] * lDT1[i, l] * lHᶻ[l, j, kl]

            dEᶻijkc_update += wJijkc * dRdX[i, j, k, 1, c] * lDT1[i, l] * lHʸ[l, j, kl]
            dEᶻijkc_update -= wJijkc * dRdX[i, j, k, 4, c] * lDT1[i, l] * lHˣ[l, j, kl]
        end

        for n = 0x1:N[2]
            dHˣijkc_update -= wJijkc * dRdX[i, j, k, 5, c] * lDT2[j, n] * lEᶻ[i, n, kl]
            dHˣijkc_update += wJijkc * dRdX[i, j, k, 8, c] * lDT2[j, n] * lEʸ[i, n, kl]

            dHʸijkc_update -= wJijkc * dRdX[i, j, k, 8, c] * lDT2[j, n] * lEˣ[i, n, kl]
            dHʸijkc_update += wJijkc * dRdX[i, j, k, 2, c] * lDT2[j, n] * lEᶻ[i, n, kl]

            dHᶻijkc_update -= wJijkc * dRdX[i, j, k, 2, c] * lDT2[j, n] * lEʸ[i, n, kl]
            dHᶻijkc_update += wJijkc * dRdX[i, j, k, 5, c] * lDT2[j, n] * lEˣ[i, n, kl]

            dEˣijkc_update += wJijkc * dRdX[i, j, k, 5, c] * lDT2[j, n] * lHᶻ[i, n, kl]
            dEˣijkc_update -= wJijkc * dRdX[i, j, k, 8, c] * lDT2[j, n] * lHʸ[i, n, kl]

            dEʸijkc_update += wJijkc * dRdX[i, j, k, 8, c] * lDT2[j, n] * lHˣ[i, n, kl]
            dEʸijkc_update -= wJijkc * dRdX[i, j, k, 2, c] * lDT2[j, n] * lHᶻ[i, n, kl]

            dEᶻijkc_update += wJijkc * dRdX[i, j, k, 2, c] * lDT2[j, n] * lHʸ[i, n, kl]
            dEᶻijkc_update -= wJijkc * dRdX[i, j, k, 5, c] * lDT2[j, n] * lHˣ[i, n, kl]
        end

        #=
        for m = 0x1:N[3]
            dHˣijkc_update -= wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lEᶻ[i, j, m, cl]
            dHˣijkc_update += wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lEʸ[i, j, m, cl]

            dHʸijkc_update -= wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lEˣ[i, j, m, cl]
            dHʸijkc_update += wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lEᶻ[i, j, m, cl]

            dHᶻijkc_update -= wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lEʸ[i, j, m, cl]
            dHᶻijkc_update += wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lEˣ[i, j, m, cl]

            dEˣijkc_update += wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lHᶻ[i, j, m, cl]
            dEˣijkc_update -= wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lHʸ[i, j, m, cl]

            dEʸijkc_update += wJijkc * dRdX[i, j, k, 9, c] * lDT3[k, m] * lHˣ[i, j, m, cl]
            dEʸijkc_update -= wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lHᶻ[i, j, m, cl]

            dEᶻijkc_update += wJijkc * dRdX[i, j, k, 3, c] * lDT3[k, m] * lHʸ[i, j, m, cl]
            dEᶻijkc_update -= wJijkc * dRdX[i, j, k, 6, c] * lDT3[k, m] * lHˣ[i, j, m, cl]
        end
        =#

        dq[i, j, k, 1, c] += invwJijkc * dHˣijkc_update
        dq[i, j, k, 2, c] += invwJijkc * dHʸijkc_update
        dq[i, j, k, 3, c] += invwJijkc * dHᶻijkc_update
        dq[i, j, k, 4, c] += invwJijkc * dEˣijkc_update
        dq[i, j, k, 5, c] += invwJijkc * dEʸijkc_update
        dq[i, j, k, 6, c] += invwJijkc * dEᶻijkc_update
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

    C = min(512 ÷ prod(size(cell)), 1)
    b = cld(size(dq)[4],C)
    S = size(cell)

    @cuda threads=(S[1], S[2]) blocks=(b, S[3]) rhs_volume_horizontal_kernel!(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(cell)),
        Val(C)
    )

    @cuda threads=(S[2], S[3]) blocks=(size(dq,4), S[1]) rhs_volume_vertical_kernel!(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(cell)),
        Val(C)
    )

    finish!(q, cm)

    J = maximum([prod(size(cell)[[idx...]]) for idx in [(1,2), (2,3), (1,3)]])
    C = 2^3# max(128 ÷ J, 1)
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

    timeend = 0.5

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

    nwarm = min(numberofsteps - 1, 10)

    #rhs!(dq, q, grid, invwJ, DT, cm)

    @info "Number of steps: $numberofsteps"
    for step = 1:numberofsteps
        if time + dt > timeend
            dt = timeend - time
        end
    
        for (i, stage) in enumerate(eachindex(RKA, RKB))
            #These should be done with CUDA.jl call to mult by scaler factor.
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

    if outputvtk 
        if rank == 0
            @info """Configuration:
                precision        = $FT
                polynomial order = $N
                array type       = $AT
            """
        end

    
        #jl # visualize solution of test problem 
        K = 4
        L = 1
        vtkdir = "vtk_semdg_maxwells_3d$(K)x$(K)x$(K)_L$(L)"
        if rank == 0
            @info """Starting Maxwells test problem with:
                ($K, $K, $K) coarse grid
                $L refinement level
            """
        end

        run(initialcondition, FT, AT, N, K, L; outputvtk = true, vtkdir, comm)
        rank == 0 && @info "Finished, vtk output written to $vtkdir"
    end

    #jl # run convergence study
    if convergetest
        numlevels = 2
        N = (4, 4, 4)
        rank == 0 && @info "Starting convergence study h-refinement"
        err = zeros(FT, numlevels)
        for l = 1:numlevels
            K = 8
            L = l - 1
            totalcells = (K * 2^L, K * 2^L)
            dofs = prod(totalcells)*prod(1 .+ N)
            err[l] = run(initialcondition, FT, AT, N, K, L; outputvtk=false, comm)

            if rank == 0
                @info @sprintf(
                    "Level %d, cells = (%2d, %2d), dof = %d, error = %.16e",
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
