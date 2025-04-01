#------------------------------Markdown Language Header-----------------------
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
#jl #FIXME: optimize kernels
#       - data trasfer kernel
#jl     - recursion kernel
#jl #FIXME: add introduction to dirac equation latex
#--------------------------------Markdown Language Header-----------------------
using Base: sign_mask, specializations
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

using MAT
using NVTX
using SpecialFunctions

import Base.abs2

const convergetest = false
const empericalconvergetest = false
const outputvtk = true
const numlevels = 3
const Lout = 2
const m = -5.0
const BC = :crbc # options: :crbc :forbc :reflect :periodic
const flux = :upwind    # options: :upwind :central
const datatype = :crbc  # options: :crbc :synthetic :none
const matcoef = :disc   # options: :disc :cont
#const f(x,y) = y > 0 ? m : -m                   # disc_h
const f(x,y) = x > 0 ? m : -m                  # disc_v
#const f(x,y) = m*(erf(10*y+3)-erf(10*y-3)-1)   # double
#const f(x,y) = m*y                             # y
#const f(x,y) = m*erf(m*y)                      # erf in y
#const f(x,y) = m*erf(m*x)                      # erf in x
const K = 16
const delta_x = 1.0
const delta_y = 1.0
const timeend = 1.2
const N = (4, 4)

BC == :crbc && @assert datatype == :crbc
datatype == :synthetic && @assert BC == :forbc

const progresswidth = 30
#const ħ = 1

function abs2(v::SVector{2,ComplexF64})
    return abs(real(conj(v)' * v))
end

@kernel function gaussian!(
    q,
    mesh,
    ::Val{N}
) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Global, NTuple)

    x = mesh[i, j, c]
    qvec = SVector{2, eltype(eltype(q))}(1,1)
    b = 0.1
        temp = exp(-(x[1]/b)^2 - (x[2]/b)^2)
    q[i, j, c] = temp*qvec
end

@kernel function exactwithinterface!(
    q,
    mesh,
    matparam,
    t,
    ::Val{N}
) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Global, NTuple)

    qvec = SVector{2, eltype(eltype(q))}(1,-1)
    mat = matparam[i, j, c]
    x = mesh[i, j, c]

    k = 10*π
    w = -k
    temp = exp(im*k*(x[1]+t)-mat*x[2])
    q[i, j, c] = temp*qvec
end

# Exact solution without interface
function exactwithoutinterface(x::SVector{2}, t)
    FT = eltype(x)
    CT = Complex{FT}
    ab = -2 * π^2
    temp = sqrt(Complex(ab - m^2))
    XY = exp(-im * π * (x[1] + x[2]))
    f1 = (1 + im) * XY * (-temp * cosh(t * temp) + im * m * sinh(t * temp)) / π
    f2 = 2 * XY * sinh(t * temp)
    return SVector{2,CT}(f1, f2)
end

# ∂ₜψ = -σ₁∂ₓψ-im(y)σ₃ψ
@kernel function crbc_tangent_kernel!(
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

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
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
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[2] * lDT2[j, n] * σ₁ * lU[i, n, c1] # -sₓDₛσ₁u
        end

        dq[i, j, c] = invwJijc * dqijc_update - im * matparamijc * σ₃ * lU[i, j, c1]
    end
end

# ∂ₜψ = dqdtan-σ₂∂ᵥψ
@kernel function crbc_volume_kernel_top!(
    dq,
    q,
    dqdtan,
    dwdn,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    @inbounds begin
        P = SA[0.5 0.5; 0.5im -0.5im]

        dq[i, j, c] += dqdtan[i, j, c] - σ₂ * P * dwdn[i, j, c]
    end
end

@kernel function crbc_volume_kernel_bottom!(
    dq,
    q,
    dqdtan,
    dwdn,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    @inbounds begin
        P = SA[0.5 0.5; -0.5im 0.5im]

        dq[i, j, c] += dqdtan[i, j, c] - σ₂ * P * dwdn[i, j, c]
    end
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

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
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

@kernel function crbc_surface_kernel!(
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

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    @inbounds if ij <= N[1]
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, c1] = zero(eltype(lqflux))
        end
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
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

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

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

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
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

@kernel function rhs_surface_kernel!(
    dq,
    q,
    data,
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

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
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

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

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

        fscale = invwJijc * wsJf

        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
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
            if BC == :crbc
                qP = data[idP]
            elseif BC == :forbc && datatype != :synthetic
                qP = (qM - σ₂ * qM) / 2
            elseif BC == :forbc && datatype == :synthetic
                qP = data[idM]
            elseif BC == :reflect
                σ₃ = SA[1.0 0.0; 0.0 -1.0]
                qP = σ₃ * qM
            end
        end

        invwJijc = invwJ[idM]

        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]

        fscale = invwJijc * wsJf
        # nx σ₁ (u_- u*) + ny σ₂ (u_- u*)
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

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
            if BC == :crbc
                qP = data[idP]
            elseif BC == :forbc && datatype != :synthetic
                qP = (qM - σ₂ * qM) / 2
            elseif BC == :forbc && datatype == :synthetic
                qP = data[idM]
            elseif BC == :reflect
                σ₃ = SA[1.0 0.0; 0.0 -1.0]
                qP = σ₃ * qM
            end
        end

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
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

function coefmap!(materialparams, gridpoints, f)
    iMax, jMax, eMax = size(gridpoints)
    if matcoef == :disc
        for e = 1:eMax
            x_temp = first.(gridpoints[[1, end], [1, end], e])[:, 1]
            y_temp = last.(gridpoints[[1, end], [1, end], e])[1, :]
            x_center = (x_temp[1] + x_temp[2]) / 2
            y_center = (y_temp[1] + y_temp[2]) / 2
            for i = 1:iMax, j = 1:jMax
                materialparams[i, j, e] = f(x_center, y_center)
            end
        end
    elseif matcoef == :cont
        for e = 1:eMax, i = 1:iMax, j = 1:jMax
            x, y = gridpoints[i,j,e]
            materialparams[i, j, e] = f(x,y)
        end
    end
end

# ∂ₜψ = -σ₁∂ₓψ-im(y)σ₃ψ
#
# P∂ₜψ = -Pσ₁∂ₓψ-im(y)Pσ₃ψ              : Bottom
#      = -Pσ₁PinvP ∂ₓψ-im(y)Pσ₃PinvPψ   : Bottom
#      =  σ₂∂ₓω-im(y)σ₁ω                : Bottom
#      =  -i∂ₓω2-im(y)ω2                : Bottom
#      =  i∂ₓω1-im(y)ω1                 : Bottom
function crbc_recursion_bottom!(dwdn, dqdtan, q, grid, m, a, sig)
    cell = referencecell(grid)
    Q = size(cell, 2) - 1
    S = size(points(grid))

    q1, q2 = components(q)
    dw1dn, dw2dn = components(dwdn)
    dq1dtan, dq2dtan = components(dqdtan)

    for e = 1:S[3]
        for x = 1:size(cell, 1)
            for (j, i) in zip(2:Q+1, Q:1)
                dw1dn[x, i, e] = (a[2*(j-1)-1] - 1) * dw1dn[x, i+1, e]

                dw1dn[x, i, e] -= a[2*(j-1)] * (dq1dtan[x, i, e] + im * dq2dtan[x, i, e])
                dw1dn[x, i, e] +=
                    a[2*(j-1)-1] * (dq1dtan[x, i+1, e] + im * dq2dtan[x, i+1, e])

                dw1dn[x, i, e] += sig[2*(j-1)] * (q1[x, i, e] + im * q2[x, i, e])
                dw1dn[x, i, e] -= sig[2*(j-1)-1] * (q1[x, i+1, e] + im * q2[x, i+1, e])

                dw1dn[x, i, e] /= (a[2*(j-1)] + 1)
            end

            dw2dn[x, 1, e] = zero(eltype(dw2dn))

            for (j, i) in zip(Q+1:-1:2, 1:Q)
                dw2dn[x, i+1, e] = (a[2*(j-1)] - 1) * dw2dn[x, i, e]

                dw2dn[x, i+1, e] -= a[2*(j-1)] * (dq1dtan[x, i, e] - im * dq2dtan[x, i, e])
                dw2dn[x, i+1, e] +=
                    a[2*(j-1)-1] * (dq1dtan[x, i+1, e] - im * dq2dtan[x, i+1, e])

                dw2dn[x, i+1, e] += sig[2*(j-1)] * (q1[x, i, e] - im * q2[x, i, e])
                dw2dn[x, i+1, e] -= sig[2*(j-1)-1] * (q1[x, i+1, e] - im * q2[x, i+1, e])

                dw2dn[x, i+1, e] /= (a[2*(j-1)-1] + 1)
            end
        end
    end

    return dwdn
end


# P∂ₜψ = -Pσ₁∂ₓψ-im(y)Pσ₃ψ          : Top
#      = -σ₂∂ₓω-im(y)σ₁ω            : Top
#      = i∂ₓω2-im(y)ω2              : Top
#      = -i∂ₓω1-im(y)ω1             : Top
function crbc_recursion_top!(dwdn, dqdtan, q, grid, m, a, sig)
    cell = referencecell(grid)
    Q = size(cell, 2) - 1
    S = size(points(grid))

    q1, q2 = components(q)
    dw1dn, dw2dn = components(dwdn)
    dq1dtan, dq2dtan = components(dqdtan)

    for e = 1:S[3]
        for x = 1:size(cell, 1)
            for j = 2:Q+1
                dw1dn[x, j, e] = (a[2*(j-1)-1] - 1) * dw1dn[x, j-1, e]

                dw1dn[x, j, e] += a[2*(j-1)] * (dq1dtan[x, j, e] - im * dq2dtan[x, j, e])
                dw1dn[x, j, e] -=
                    a[2*(j-1)-1] * (dq1dtan[x, j-1, e] - im * dq2dtan[x, j-1, e])

                dw1dn[x, j, e] += sig[2*(j-1)] * (q1[x, j, e] - im * q2[x, j, e])
                dw1dn[x, j, e] -= sig[2*(j-1)-1] * (q1[x, j-1, e] - im * q2[x, j-1, e])

                dw1dn[x, j, e] /= (a[2*(j-1)] + 1)
            end

            dw2dn[x, Q+1, e] = zero(eltype(dw2dn))

            for j = Q+1:-1:2
                dw2dn[x, j-1, e] = (a[2*(j-1)] - 1) * dw2dn[x, j, e]

                dw2dn[x, j-1, e] += a[2*(j-1)] * (dq1dtan[x, j, e] + im * dq2dtan[x, j, e])
                dw2dn[x, j-1, e] -=
                    a[2*(j-1)-1] * (dq1dtan[x, j-1, e] + im * dq2dtan[x, j-1, e])

                dw2dn[x, j-1, e] += sig[2*(j-1)] * (q1[x, j, e] + im * q2[x, j, e])
                dw2dn[x, j-1, e] -= sig[2*(j-1)-1] * (q1[x, j-1, e] + im * q2[x, j-1, e])

                dw2dn[x, j-1, e] /= (a[2*(j-1)-1] + 1)
            end
        end
    end

    return dwdn
end

function crbc_rhs!(dq, q, dwdn, dqdtan, grid, invwJ, DT, materialparams, a, sig, cm; orient = "top")
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    C = max(512 ÷ prod(size(cell)), 1)
    crbc_tangent_kernel!(backend, (size(cell)..., C))(
        dqdtan,
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

    J = maximum(size(cell))
    C = max(128 ÷ J, 1)
    crbc_surface_kernel!(backend, (J, C))(
        dqdtan,
        q,
        fm.vmapM,
        fm.vmapP,
        n,
        wsJ,
        invwJ,
        Val(size(cell)),
        Val(C);
        ndrange = (J, last(size(dq))),
    )

    if orient == "top"
        #FIXME: AHHHHH this is real bad
        #dwdn = crbc_recursion_top!(dwdn, dqdtan, q, grid, m, a, sig)
        #FIXME: end

        C = max(512 ÷ prod(size(cell)), 1)
        crbc_volume_kernel_top!(backend, (size(cell)..., C))(
            dq,
            q,
            dqdtan,
            dwdn,
            Val(size(cell)),
            Val(C);
            ndrange = size(dq),
        )
    elseif orient == "bottom"
        #FIXME: AHHHHH this is real bad
        #dwdn = crbc_recursion_bottom!(dwdn, dqdtan, q, grid, m, a, sig)
        #FIXME: end

        C = max(512 ÷ prod(size(cell)), 1)
        crbc_volume_kernel_bottom!(backend, (size(cell)..., C))(
            dq,
            q,
            dqdtan,
            dwdn,
            Val(size(cell)),
            Val(C);
            ndrange = size(dq),
        )
    end
end


function rhs!(dq, q, grid, data, invwJ, DT, materialparams, bc, cm)
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
        data,
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
    FT,
    AT,
    N,
    K,
    L;
    outputvtk = false,
    progress = true,
    vtkdir = "output",
    comm = MPI.COMM_WORLD,
)
    rank = MPI.Comm_rank(comm)
    cell = LobattoCell{FT,AT}((N .+ 1)...)
    coordinates = (
        range(FT(-delta_x), stop = FT(delta_x), length = K + 1),
        range(FT(-delta_y), stop = FT(delta_y), length = K + 1),
    )

    if BC == :periodic
        periodicity = (true, true)
    else
        periodicity = (true, false)
    end
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(gm)


    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates))
    dt = cfl * dx / (maximum(N)+1)^2

    numberofsteps = ceil(Int, timeend / dt)
    energy = []
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

    energy_output = function (step, energy, q, grid)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            _, _, wJ = components(first(volumemetrics(grid)))

            e = sqrt(sum(Adapt.adapt(Array, wJ .* abs2.(q))))

            push!(energy, e)
        end
    end

    do_output = function (step, time, q, matparam, grid)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            cell = referencecell(grid)
            cd(vtkdir) do
                filename = "step$(lpad(step, 6, '0'))"
                vtkfile = vtk_grid(filename, grid)
                P = toequallyspaced(cell)
                vtkfile["|q|²"] = Adapt.adapt(Array, P * abs2.(q))
               vtkfile["m"] = Adapt.adapt(Array, matparam)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end


    #jl # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)
    #jl # precompute derivative transpose
    DT = derivatives_1d(cell)

    fn(x, y) = f(x,delta_y)
    fs(x, y) = f(x,-delta_y)
    fe(x, y) = f(delta_x,y)
    fw(x, y) = f(-delta_x, y)

    materialparams_H = zeros(size(wJ))
    gridpoints_H = adapt(Array, grid.points)

    coefmap!(materialparams_H, gridpoints_H, f)
    materialparams = adapt(AT, materialparams_H)

    #jl # initialize state
    q = GridArray{SVector{2, ComplexF64}}(undef, grid)

    backend = Raven.get_backend(q)
    if convergetest == true
        exactwithinterface!(backend, (size(cell)..., 1))(
            q,
            points(grid),
            materialparams,
            0.0,
            Val(size(cell));
            ndrange = size(q)
        )
    else
        gaussian!(backend, (size(cell)..., 1))(
            q,
            points(grid),
            Val(size(cell));
            ndrange = size(q)
        )
    end
    #jl # storage for RHS
    dq = KernelAbstractions.zeros(backend, eltype(q), size(q))

    # adjust boundary code
    bc_H = adapt(Array, boundarycodes(grid))
    for j = 1:size(bc_H, 2)
        if last(gridpoints_H[end, end, j]) ≈ delta_y  #North
            bc_H[4, j] = 2
        end

        if last(gridpoints_H[1, 1, j]) ≈ -delta_y # South
            bc_H[3, j] = 2
        end
    end

    bc = adapt(AT, bc_H)
    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #%%%%%%%%%%%%%#
    #  CRBC DATA  #
    #%%%%%%%%%%%%%#
    data = similar(q)
    data .= Ref(zero(eltype(q)))
    if BC == :crbc
        Q = 10
        crbc_cell = LobattoCell{FT,AT}(N[1] + 1, Q + 1)

        data .= Ref(zero(eltype(q)))

        # TOP
        crbc_coordinates_top = (
            range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
            range(FT(delta_y), stop = FT(delta_y + 1.0), length = 2),
        )
        crbc_gm_top = GridManager(
            crbc_cell,
            brick(crbc_coordinates_top, (true, false));
            comm = comm,
            min_level = 0,
        )
        crbc_grid_top = generate(crbc_gm_top)
        crbc_q_top = GridArray{eltype(q)}(undef, crbc_grid_top)
        crbc_q_top .= Ref(zero(eltype(q)))
        crbc_dq_top = similar(crbc_q_top)
        crbc_dwdn_top = similar(crbc_q_top)
        crbc_dqdtan_top = similar(crbc_q_top)
        crbc_dq_top .= Ref(zero(eltype(q)))
        crbc_dwdn_top .= Ref(zero(eltype(q)))
        crbc_dqdtan_top .= Ref(zero(eltype(q)))

        _, _, crbc_wJ_top = components(first(volumemetrics(crbc_grid_top)))
        crbc_invwJ_top = inv.(crbc_wJ_top)
        crbc_DT_top = derivatives_1d(crbc_cell)

        #FIXME: AHHHHH this is real bad
        crbc_interface_top = findall(≈(delta_y), adapt(Array, last.(points(crbc_grid_top))))
        bulk_interface_top = findall(≈(delta_y), adapt(Array, last.(points(grid))))
        #FIXME: end

        # BOTTOM
        crbc_coordinates_bottom = (
            range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
            range(FT(-delta_y - 1.0), stop = FT(-delta_y), length = 2),
        )
        crbc_gm_bottom = GridManager(
            crbc_cell,
            brick(crbc_coordinates_bottom, (true, false));
            comm = comm,
            min_level = 0,
        )
        crbc_grid_bottom = generate(crbc_gm_bottom)
        crbc_q_bottom = GridArray{eltype(q)}(undef, crbc_grid_bottom)
        crbc_q_bottom .= Ref(zero(eltype(q)))
        crbc_dq_bottom = copy(crbc_q_bottom)
        crbc_dwdn_bottom = copy(crbc_q_bottom)
        crbc_dqdtan_bottom = copy(crbc_q_bottom)
        _, _, crbc_wJ_bottom = components(first(volumemetrics(crbc_grid_bottom)))
        crbc_invwJ_bottom = inv.(crbc_wJ_bottom)
        crbc_DT_bottom = derivatives_1d(crbc_cell)

        # crbc parameters
        _, _, crbc_wJ_top = components(first(volumemetrics(crbc_grid_top)))
        crbc_materialparams_top_H = zeros(size(crbc_wJ_top))
        crbc_gridpoints_top_H = adapt(Array, crbc_grid_top.points)
        coefmap!(crbc_materialparams_top_H, crbc_gridpoints_top_H, fn)
        crbc_materialparams_top = adapt(AT, crbc_materialparams_top_H)

        _, _, crbc_wJ_bottom = components(first(volumemetrics(crbc_grid_bottom)))
        crbc_materialparams_bottom_H = zeros(size(crbc_wJ_bottom))
        crbc_gridpoints_bottom_H = adapt(Array, crbc_grid_bottom.points)
        coefmap!(crbc_materialparams_bottom_H, crbc_gridpoints_bottom_H, fs)
        crbc_materialparams_bottom = adapt(AT, crbc_materialparams_bottom_H)

        #FIXME: AHHHHH this is real bad
        crbc_interface_bottom = findall(≈(-delta_y), adapt(Array, last.(points(crbc_grid_bottom))))
        bulk_interface_bottom = findall(≈(-delta_y), adapt(Array, last.(points(grid))))
        #FIXME: end
    end # crbc initialization

    #jl # initial output
    step = 0
    time = FT(0)

    matlabdata = MAT.matread("optimal_cosines_data.mat")
    a = matlabdata["a"]
    sig = matlabdata["sig"]

    do_output(step, time, q, materialparams, grid)

    progress_stepwidth = cld(numberofsteps, progresswidth)
    elapsed_time = @elapsed begin
        for step = 1:numberofsteps
            if rank == 0 && progress && mod(step, progress_stepwidth) == 0
                print(
                    "\r" *
                    raw"-\|/"[cld(step, progress_stepwidth)%4+1] *
                    "="^cld(step, progress_stepwidth) *
                    " "^(progresswidth - cld(step, progress_stepwidth)) *
                    "|",
                )
                @printf "%.1f%%" step / numberofsteps * 100
            end

            if time + dt > timeend
                dt = timeend - time
            end

            if BC == :crbc
                crbc_dwdn_top_h = Adapt.adapt(Array, crbc_dwdn_top)
                crbc_dwdn_bottom_h = Adapt.adapt(Array, crbc_dwdn_bottom)
            end

            data_h = Adapt.adapt(Array, data)
            q_h = Adapt.adapt(Array,q)

            for stage in eachindex(RKA, RKB)
                @. dq *= RKA[stage]
                if BC == :crbc
                    @. crbc_dq_top *= RKA[stage]
                    @. crbc_dq_bottom *= RKA[stage]

                    #FIXME: AHHHHH this is real bad
                    P = SA[1 -im; 1 im]
                    crbc_dwdn_top_h[crbc_interface_top] .=
                        [P * i for i in q_h[bulk_interface_top]]

                    P = SA[-1 -im; -1 im]
                    crbc_dwdn_bottom_h[crbc_interface_bottom] .=
                        [P * i for i in q_h[bulk_interface_bottom]]

                    crbc_dwdn_top = Adapt.adapt(AT, crbc_dwdn_top_h)
                    crbc_dwdn_bottom = Adapt.adapt(AT, crbc_dwdn_bottom_h)
                    #FIXME: end
                    crbc_rhs!(
                        crbc_dq_top,
                        crbc_q_top,
                        crbc_dwdn_top,
                        crbc_dqdtan_top,
                        crbc_grid_top,
                        crbc_invwJ_top,
                        crbc_DT_top,
                        crbc_materialparams_top,
                        a,
                        sig,
                        cm;
                        orient = "top",
                    )

                    crbc_rhs!(
                        crbc_dq_bottom,
                        crbc_q_bottom,
                        crbc_dwdn_bottom,
                        crbc_dqdtan_bottom,
                        crbc_grid_bottom,
                        crbc_invwJ_bottom,
                        crbc_DT_bottom,
                        crbc_materialparams_bottom,
                        a,
                        sig,
                        cm;
                        orient = "bottom",
                    )

                    #FIXME: AHHHH this is real bad
                    crbc_dwdn_top_h  = Adapt.adapt(Array, crbc_dwdn_top)
                    crbc_dwdn_bottom_h = Adapt.adapt(Array, crbc_dwdn_bottom)

                    Pinv = SA[0.5 0.5; 0.5*im -0.5*im]
                    data_h[bulk_interface_top] .=
                        [Pinv * i for i in crbc_dwdn_top_h[crbc_interface_top]]

                    Pinv = SA[-0.5 -0.5; 0.5*im -0.5*im]
                    data_h[bulk_interface_bottom] .=
                        [Pinv * i for i in crbc_dwdn_bottom_h[crbc_interface_bottom]]
                    #FIXME: end
                end


                if datatype == :synthetic
                    exactwithinterface!(backend, (size(cell)..., 1))(
                        data,
                        points(grid),
                        materialparams,
                        time,
                        Val(size(cell));
                        ndrange = size(q)
                    )
                end

                rhs!(dq, q, grid, data, invwJ, DT, materialparams, bc, cm)

                @. q += RKB[stage] * dt * dq

                if BC == :crbc
                    @. crbc_q_top += RKB[stage] * dt * crbc_dq_top
                    @. crbc_q_bottom += RKB[stage] * dt * crbc_dq_bottom
                end
            end
            time += dt

            #do_output(step, time, q, grid)
            do_output(step, time, q, materialparams, grid)
            energy_output(step, energy, q, grid)
        end # time step for loop
    end # elapsed time

    if rank == 0
        println(
            "\r" *
            raw"-\|/"[cld(step, progress_stepwidth)%4+1] *
            "="^progresswidth *
            "|100.0% | runtime: $(elapsed_time) sec.",
        )
    end

    do_output(numberofsteps, timeend, q, materialparams, grid)
    energy_output(numberofsteps, energy, q, grid)

    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end

    if convergetest
        # compute error
        _, _, wJ = components(first(volumemetrics(grid)))
        qexact = similar(q)

        exactwithinterface!(backend, (size(cell)..., 1))(
            qexact,
            points(grid),
            materialparams,
            timeend,
            Val(size(cell));
            ndrange = size(q)
        )

        #jl # TODO add sum to GridArray so the following reduction is on the device
        err = sqrt(sum(Adapt.adapt(Array, wJ .* abs2.(q .- qexact))))

        return q, grid, energy, err
    end

    return q, grid, energy, nothing
end

let
    FT = Float64
    AT = !(BC == :crbc) && CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array
    #AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
        CUDA.allowscalar(true) #FIXME: AHHHH need this for recursion!!
    end

    vtkdir = "vtk_semidg_dirac_2d$(K)x$(K)_L$(Lout)_$(String(BC))_$(timeend)"

    if rank == 0
        configdata = """Configuration:
            precision           = $FT
            array type          = $AT

            convergence test    = $convergetest
            emperical conv test = $empericalconvergetest
            conv numlevels      = $(empericalconvergetest || convergetest ? numlevels : "N/A")
            outputvtk           = $outputvtk
            outputdir           = $(outputvtk ? vtkdir : "N/A")
            base                = $K
            level refine        = $Lout
            polynomial order    = $N
            flux type           = $(String(flux))
            matparam            = $(String(matcoef))
            m                   = $m
            bc type             = periodic x $(String(BC))
            ghost ∂Ω data type  = $(String(datatype))
            domain width        = $(2*delta_x)
            domain height       = $(2*delta_y)
            time end            = $(timeend)
        """

        @info configdata
        if outputvtk
            if !isdir(vtkdir)
                mkpath(vtkdir)
            end
            cd(vtkdir) do
                open("config.txt", "w+") do file
                    write(file, configdata)
                end
            end
        end
    end

    if outputvtk
        #jl # visualize solution of dirac wave
        if rank == 0
            @info """Starting Dirac solver with:
                ($K, $K) coarse grid
                $Lout refinement level
            """
        end

        _, _, energy, _ = run(
            FT,
            AT,
            N,
            K,
            Lout;
            outputvtk = outputvtk,
            progress = true,
            vtkdir,
            comm,
        )
        cd(vtkdir) do
            open("energy.txt", "w") do file
                e0 = energy[1]
                for idx = 1:length(energy)
                    println(file, energy[idx] / e0)
                end
            end
        end
        rank == 0 && @info "Finished, vtk output written to $vtkdir"
    end # outputvtk

    if convergetest
        rank == 0 && @info "Starting convergence study h-refinement"
        err = zeros(FT, numlevels)
        @assert m > 0 "Exact solution requires m > 0"

        for l = 1:numlevels
            L = l - 1
            totalcells = (K * 2^L, K * 2^L)
            dofs = prod(totalcells) * prod(1 .+ N)
            _, _, _, err[l] = run(
                FT,
                AT,
                N,
                K,
                L;
                outputvtk = outputvtk,
                progress = true,
                vtkdir,
                comm,
            )

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
            convdata =
                "Convergence rates against exact solution:\n" * join(
                    ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(numlevels-1)],
                    "\n",
                )

            @info convdata
            if outputvtk
                cd(vtkdir) do
                    open("config.txt", "a") do file
                        write(file, "----------------")
                        write(file, convdata)
                        write(file, "\n")
                    end
                end
            end
        end
    end # converge test

    if empericalconvergetest && rank == 0
        cell = LobattoCell{FT,AT}((N .+ 1)...)
        (a, b) = tohalves_1d(cell)

        tohalves_q1 = Raven.Kron((b[1], a[1]))
        tohalves_q2 = Raven.Kron((b[1], a[2]))
        tohalves_q3 = Raven.Kron((b[2], a[1]))
        tohalves_q4 = Raven.Kron((b[2], a[2]))

        rank == 0 && @info "Starting Emperical convergence study h-refinement"
        err = zeros(FT, numlevels)

        for l = 1:numlevels
            L = l - 1
            totalcells = (K * 2^L, K * 2^L)
            dofs = prod(totalcells) * prod(1 .+ N)
            final_course, _, _, _ = run(FT, AT, N, K, L; outputvtk = false, comm)
            final_fine, grid, _, _ = run(FT, AT, N, K, L + 1; outputvtk = false, comm)

            _, _, wJ = components(first(volumemetrics(grid)))

            fp_h = similar(final_fine)

            c = size(final_fine, 3)
            fp_h[:, :, 1:4:c] = tohalves_q1 * final_course
            fp_h[:, :, 2:4:c] = tohalves_q2 * final_course
            fp_h[:, :, 3:4:c] = tohalves_q3 * final_course
            fp_h[:, :, 4:4:c] = tohalves_q4 * final_course

            if outputvtk
                rank == 0 && mkpath(vtkdir)
                pvd = rank == 0 ? paraview_collection("conv") : nothing

                do_output = function (L, time, q1, q2, grid)
                    if outputvtk
                        cell = referencecell(grid)
                        cd(vtkdir) do
                            filename = "Level$(lpad(L, 2, '0'))"
                            vtkfile = vtk_grid(filename, grid)
                            P = toequallyspaced(cell)
                            vtkfile["|e|²"] = Adapt.adapt(Array, P * abs2.(q1 .- q2))
                            vtk_save(vtkfile)
                            if rank == 0
                                pvd[time] = vtkfile
                            end
                        end
                    end
                end

                do_output(L, 0.0, final_fine, fp_h, grid)
            end

            err[l] = sqrt(abs(sum(Adapt.adapt(Array, wJ .* abs2.(fp_h .- final_fine)))))

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
            convdata =
                "Emperical Convergence rates:\n" * join(
                    ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(numlevels-1)],
                    "\n",
                )

            @info convdata
            if outputvtk
                cd(vtkdir) do
                    open("config.txt", "a") do file
                        write(file, "----------------")
                        write(file, convdata)
                        write(file, "\n")
                    end
                end
            end
        end
    end # emperical convergence test
end
