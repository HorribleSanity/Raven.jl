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
#   #FIXME: (!) compute error on GPU
#   #FIXME: is async Kernel launch hurting me??
#   #FIXME: (!) stupid thread count convention
#   #FIXME: homaginize recursion kernels
#   #FIXME: reduce crbc memory model
#   #FIXME: (!) bounds check unsafe_indices & inline everything & inbounds
#   #FIXME: cli flags
#   #FIXME: realify dirac
#   #FIXME: (!) num dofs in report
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
using SpecialFunctions

#https://github.com/guiyrt/MLinJulia/blob/5b0cf79258848e2723dc270b550066cac07af5a9/src/calodiffusion/profile_cuda.jl#L3
import Base.abs2

const convergetest = false
const empericalconvergetest = false
const outputvtk = true
const bigrun = true
const numlevels = 4
const Lout = 0
const m = -35.0 # -70.0
const BC = :crbc_lr # options:crbc_tb :crbc_lr :forbc :reflect :periodic
const flux = :upwind    # options: :upwind :central
const datatype = :crbc  # options: :crbc :synthetic :none
const matcoef = :smooth  # options: :constant :disc :smooth
#const f(x, y) = y > 0 ? m : -m                   # disc_h
#const f(x,y) = x > 0 ? m : -m                  # disc_v
#const f(x,y) = -m*(erf(m*y+5)-erf(m*y)+erf(m*y-5))   # double
#const f(x, y) = m * y                             # y
const f(x, y) = m * x
#const f(x,y) = voidscatter(x,y)
#const f(x,y) = m*erf(m*y)                      # erf in y
#const f(x,y) = m*erf(m*x)                      # erf in x
const K = 16
const delta_x = 1.0
const delta_y = 1.0
const timeend = 1.0
const polydegree = 4
const N = (polydegree, polydegree)

function voidscatter(x, y)
    r = delta_x / 8
    p = sqrt(x^2 + y^2)
    if r < p
        if abs(y) < p - r
            return m * y
        else
            return m * (y / abs(y)) * (p - r)
        end
    else
        return 0.0
    end
end

(BC == :crbc_tb || BC == :crbc_lr) && @assert datatype == :crbc
datatype == :synthetic && @assert BC == :forbc

const progresswidth = 30
#const ħ = 1

function abs2(v::SVector{2,ComplexF64})
    return abs(real(v' * v))
end

@kernel function gaussian!(q, mesh, ::Val{N}) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Group, NTuple)

    x = mesh[i, j, c]
    qvec = SVector{2,eltype(eltype(q))}(1, -im) # vert (1, -im), horz (1, 1)
    a = 0.0
    temp = exp(-abs(m) * ((x[1] + a)^2) / 2 - abs(m) * (x[2]^2) / 2)
    q[i, j, c] = temp * qvec
end

@kernel function exactwithinterface!(q, mesh, matparam, t, ::Val{N}) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Global, NTuple)

    qvec = SVector{2,eltype(eltype(q))}(1, -1)
    mat = matparam[i, j, c]
    x = mesh[i, j, c]

    k = 10 * π
    w = -k
    temp = exp(im * k * (x[1] + t) - mat * x[2])
    q[i, j, c] = temp * qvec
end

@kernel function exactwithoutinterface!(q, mesh, matparam, t, ::Val{N}) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Global, NTuple)

    qvec = SVector{2,eltype(eltype(q))}(1, -1)
    mat = matparam[i, j, c]
    x = mesh[i, j, c]

    q[i, j, c] = exactwithoutinterface(x, t)
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

@kernel function idxcopyP!(
    qto,
    qto_idx,
    qfrom,
    qfrom_idx,
    P,
)
    i = @index(Global, Linear)

    if i <= length(qto_idx)
        qto[qto_idx[i]] = P * qfrom[qfrom_idx[i]]
    end
end

# ∂ₜψ = -σ₁∂ₓψ-im(y)σ₃ψ
@kernel function crbc_tangent_tb!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    materialparams,
    ::Val{N},
    ::Val{C},
    ::Val{S}
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg - 1) * C + cl

    # C compile time data corrisponding to cells per workgroup
    lDT1 = @localmem eltype(eltype(q)) (N[1], N[1])
    lDT2 = @localmem eltype(eltype(q)) (N[2], N[2])
    lU = @localmem eltype(q) (N..., C) # local solution at prod(N) dofs in C cells of domain

    # Pauli Matrices: SA denotes StaticArrays
    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]

    if c <= S
        for sj = 0x0:N[2]:(N[1]-0x1) # Sets stride in j
            if j + sj <= N[1] && cl == 0x1 # localmemory is shared amongst a workgroup so only cl needs to work
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && cl == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        # data (i, j) in global index cell c
        qijc = q[i, j, c]
        lU[i, j, cl] = qijc
    end

    @synchronize

    c = (cg - 1) * C + cl

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]

    if c <= S
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]
        wJdRdXijc = wJijc * dRdXijc
        matparamijc = materialparams[i, j, c]

        @unroll for m = 0x1:N[1]
            dqijc_update -= wJdRdXijc[1] * lDT1[i, m] * σ₁ * lU[m, j, cl] # -rₓDᵣσ₁u
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[2] * lDT2[j, n] * σ₁ * lU[i, n, cl] # -sₓDₛσ₁u
        end

        dq[i, j, c] = invwJijc * dqijc_update - im * matparamijc * σ₃ * lU[i, j, cl]
    end
end

# ∂ₜψ = -σ₂∂ᵥψ -im(y)σ₃ψ
@kernel inbounds=true unsafe_indices=true function crbc_tangent_lr!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    materialparams,
    ::Val{N},
    ::Val{C},
    ::Val{S}
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg - 1) * C + cl

    # C compile time data corrisponding to cells per workgroup
    lDT1 = @localmem eltype(eltype(q)) (N[1], N[1])
    lDT2 = @localmem eltype(eltype(q)) (N[2], N[2])
    lU = @localmem eltype(q) (N..., C) # local solution at prod(N) dofs in C cells of domain

    if c <= S
        # Pauli Matrices: SA denotes StaticArrays
        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]
        σ₃ = SA[1.0 0.0; 0.0 -1.0]

        for sj = 0x0:N[2]:(N[1]-0x1) # Sets stride in j
            if j + sj <= N[1] && cl == 0x1 # localmemory is shared amongst a workgroup so only c1 needs to work
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && cl == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        # data (i, j) in global index cell c
        qijc = q[i, j, c]
        lU[i, j, cl] = qijc
    end

    @synchronize

    c = (cg - 1) * C + cl

    if c <= S
        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]
        σ₃ = SA[1.0 0.0; 0.0 -1.0]
    # Here a thread will compute the (i,j)th component of the dq matrix where i,j is the dof and c is the global cell index
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]
        wJdRdXijc = wJijc * dRdXijc
        matparamijc = materialparams[i, j, c]


        @unroll for m = 0x1:N[1]
            dqijc_update -= wJdRdXijc[3] * lDT1[i, m] * σ₂ * lU[m, j, cl] # -rᵥDᵣσ₂u
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[4] * lDT2[j, n] * σ₂ * lU[i, n, cl] # -sᵥDₛσ₂u
        end

        dq[i, j, c] = invwJijc * dqijc_update - im * matparamijc * σ₃ * lU[i, j, cl]
    end
end


# ∂ₜψ = dqdtan-σ₂∂ᵥψ
@kernel inbounds=true unsafe_indices=true function crbc_volume_top!(
    dq,
    q,
    dqdtan,
    dwdn,
    ::Val{N},
    ::Val{C},
    ::Val{S},
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    P = SA[0.5 0.5; 0.5im -0.5im]
    c = (cg - 1) * C + cl

    if c <= S
        dq[i, j, c] += dqdtan[i, j, c] - σ₂ * P * dwdn[i, j, c]
    end
end

@kernel inbounds=true unsafe_indices=true function crbc_volume_bottom!(
    dq,
    q,
    dqdtan,
    dwdn,
    ::Val{N},
    ::Val{C},
    ::Val{S}
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    P = SA[0.5 0.5; -0.5im 0.5im]
    c = (cg - 1) * C + cl

    if c <= S
        dq[i, j, c] += dqdtan[i, j, c] - σ₂ * P * dwdn[i, j, c]
    end
end

@kernel function crbc_volume_left!(
    dq,
    q,
    dqdtan,
    dwdn,
    ::Val{N},
    ::Val{C},
    ::Val{S}
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg - 1) * C + cl

    if c <= S
        σ₁ = SA[0.0 1.0; 1.0 0.0]
        P = SA[0.5 0.5; -0.5 0.5]

        dq[i, j, c] += dqdtan[i, j, c] - σ₁ * P * dwdn[i, j, c]
    end
end


@kernel inbounds=true unsafe_indices=true function crbc_volume_right!(
    dq,
    q,
    dqdtan,
    dwdn,
    ::Val{N},
    ::Val{C},
    ::Val{S},
) where {N,C,S}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    c = @index(Group, Linear) # global cell numbering

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    if c <= S
        P = SA[0.5 0.5; 0.5 -0.5]

        dq[i, j, c] += dqdtan[i, j, c] - σ₁ * P * dwdn[i, j, c]
    end
end



# ∂ₜψ = -σ₁∂ₓψ-σ₂∂ᵥψ-im(y)σ₃ψ
@kernel inbounds=true unsafe_indices=true function rhs_volume!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    materialparams,
    ::Val{N},
    ::Val{C},
    ::Val{S},
) where {N,C,S}
    i, j, cl  = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg-1) * C + cl

    lDT1 = @localmem eltype(eltype(q)) (N[1], N[1])
    lDT2 = @localmem eltype(eltype(q)) (N[2], N[2])
    lU = @localmem eltype(q) (N..., C) # local solution at prod(N) dofs in C cells of domain

    if c <= S
        # Pauli Matrices: SA denotes StaticArrays
        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]
        σ₃ = SA[1.0 0.0; 0.0 -1.0]

        for sj = 0x0:N[2]:(N[1]-0x1) # Sets stride in j
            if j + sj <= N[2] && cl == 0x1 # localmemory is shared amongst a workgroup so only cl needs to work
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && cl == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

            # data (i, j) in global index cell c
        qijc = q[i, j, c]
        lU[i, j, cl] = qijc
    end

    @synchronize
    c = (cg-1) * C + cl

    if c <= S
        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]
        σ₃ = SA[1.0 0.0; 0.0 -1.0]
        # Here a thread will compute the (i,j)th component of the dq matrix where i,j is the dof and c is the global cell index
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]
        wJdRdXijc = wJijc * dRdXijc
        matparamijc = materialparams[i, j, c]


        @unroll for m = 0x1:N[1]
            dqijc_update -= wJdRdXijc[1] * lDT1[i, m] * σ₁ * lU[m, j, cl] # -rₓDᵣσ₁u
            dqijc_update -= wJdRdXijc[3] * lDT1[i, m] * σ₂ * lU[m, j, cl] # -rᵥDᵣσ₂u
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[2] * lDT2[j, n] * σ₁ * lU[i, n, cl] # -sₓDₛσ₁u
            dqijc_update -= wJdRdXijc[4] * lDT2[j, n] * σ₂ * lU[i, n, cl] # -sᵥDₛσ₂u
        end
        dq[i, j, c] += invwJijc * dqijc_update - im * matparamijc * σ₃ * lU[i, j, cl]
    end
end

@kernel inbounds=true unsafe_indices=true function crbc_surface_tb!(
    dq,
    q,
    vmapM,
    vmapP,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
    ::Val{S}
) where {N,C,S}
    ij, cl = @index(Local, NTuple)
    cg = @index(Group, Linear)
    lqflux = @localmem eltype(dq) (N..., C)
    c = (cg - 1) * C + cl

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    if ij <= N[1] && c <= S
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, cl] = zero(eltype(lqflux))
        end
    end

    @synchronize

    c = (cg - 1) * C + cl
    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    if ij <= N[2] && c <= S
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
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

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
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    # RHS update
    i = ij
    c = (cg - 1) * C + cl
    if i <= N[1] && c <= S
        @unroll for j = 1:N[2]
            dq[i, j, c] += lqflux[i, j, cl]
        end
    end
end

#TODO: clean this
@kernel inbounds=true unsafe_indices=true function crbc_surface_lr!(
    dq,
    q,
    vmapM,
    vmapP,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
    ::Val{S}
) where {N,C,S}
    ij, cl = @index(Local, NTuple)
    cg = @index(Group, Linear)
    lqflux = @localmem eltype(dq) (N..., C)
    c = (cg - 1) * C + cl

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    if ij <= N[1] && c <= S
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, cl] = zero(eltype(lqflux))
        end
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    c = (cg - 1) * C + cl
    if ij <= N[1] && c <= S
        # Faces with s=-1 : South
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

        fscale = invwJijc * wsJf
        # nx σ₁ (u_- u*) + ny σ₂ (u_- u*)
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

        # Faces with s=1 : North
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

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        if flux == :upwind
            numflux = (qM - qP + A * (qM + qP)) / 2
        else
            numflux = nf[1] * σ₁ * (qM + qP) / 2 + nf[2] * σ₂ * (qM + qP) / 2
        end
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    # RHS update
    i = ij
    c = (cg - 1) * C + cl
    if i <= N[1] && c <= S
        @unroll for j = 1:N[2]
            dq[i, j, c] += lqflux[i, j, cl]
        end
    end
end


@kernel inbounds=true unsafe_indices=true function rhs_surface!(
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
    ::Val{S}
) where {N,C,S}
    ij, cl = @index(Local, NTuple)
    cg = @index(Group, Linear)
    lqflux = @localmem eltype(dq) (N..., C)
    @infiltrate
    c = (cg - 1) * C + cl

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    if ij <= N[1] && c <= S
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, cl] = zero(eltype(lqflux))
        end
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    c = (cg - 1) * C + cl
    if ij <= N[2] && c <= S
        # Faces with r=-1 : West
        i = 1
        j = ij
        fid = j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = bc[1, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 3
            if BC == :crbc_lr
                qP = data[idP]
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
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

        # Faces with r=1 : East
        i = N[1]
        j = ij
        fid = N[2] + j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = bc[2, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 3
            if BC == :crbc_lr
                qP = data[idP]
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
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    c = (cg - 1) * C + cl
    if ij <= N[1] && c <= S
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
            if BC == :crbc_tb
                qP = data[idP]
            elseif BC == :forbc && datatype != :synthetic
                qP = (qM - σ₂ * qM) / 2
            elseif BC == :forbc && datatype == :synthetic
                qP = data[idM]
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
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

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
            if BC == :crbc_tb
                qP = data[idP]
            elseif BC == :forbc && datatype != :synthetic
                qP = (qM - σ₂ * qM) / 2
            elseif BC == :forbc && datatype == :synthetic
                qP = data[idM]
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
        lqflux[i, j, cl] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    # RHS update
    i = ij
    c = (cg - 1) * C + cl
    if i <= N[1] && c <= S
        @unroll for j = 1:N[2]
            dq[i, j, c] += lqflux[i, j, cl]
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
    elseif matcoef == :smooth
        for e = 1:eMax, i = 1:iMax, j = 1:jMax
            x, y = gridpoints[i, j, e]
            materialparams[i, j, e] = f(x, y)
        end
    elseif matcoef == :constant
        for e = 1:eMax, i = 1:iMax, j = 1:jMax
            materialparams[i, j, e] = m
        end
    end
end

@kernel inbounds=true unsafe_indices=true function crbc_recursion_bottom!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    m,
    a,
    sig,
    ::Val{I},
    ::Val{Q},
) where {I, Q}
    i = @index(Local, Linear)
    c = @index(Group, Linear)

    if i <= I
        @unroll for j = Q:1
            aodd = a[2*(j-1)-1]
            aeven = a[2*(j-1)]

            dw1dn_ijc = (aodd - 1) * dw1dn[i, j+1, c]
            dw1dn_ijc -= aeven * (dq1dtan[i, j, c] + im * dq2dtan[i, j, c])
            dw1dn_ijc += aodd * (dq1dtan[i, j+1, c] + im * dq2dtan[i, j+1, c])
            dw1dn_ijc += sig[2*(j-1)] * (q1[i, j, c] + im * q2[i, j, c])
            dw1dn_ijc -= sig[2*(j-1)-1] * (q1[i, j+1, c] - im * q2[i, j+1, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end

        dw2dn[i, 1, c] = zero(eltype(dw2dn))

        @unroll for j = 1:Q
            pidx = Q + 2 - j
            aodd = a[2*(pidx-1)-1]
            aeven = a[2*(pidx-1)]
            sigeven = sig[2*(pidx-1)]
            sigodd = sig[2*(pidx-1)-1]

            dw2dn_ijc = (aodd - 1) * dw2dn[i, j, c]
            dw2dn_ijc -= aeven * (dq1dtan[i, j, c] - im * dq2dtan[i, j, c])
            dw2dn_ijc += aodd * (dq1dtan[i, j+1, c] - im * dq2dtan[i, j+1, c])
            dw2dn_ijc += sigeven * (q1[i, j, c] - im * q2[i, j, c])
            dw2dn_ijc -= sigodd * (q1[i, j+1, c] - im * q2[i, j+1, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i, j+1, c] = dw2dn_ijc
        end
    end
end

@kernel function crbc_recursion_top!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    m,
    a,
    sig,
    ::Val{I},
    ::Val{Q},
) where {I, Q}
    i = @index(Local, Linear)
    c = @index(Group, Linear)

    if i <= I
        @unroll for j = 2:Q+1
            aodd = a[2*(j-1)-1]
            aeven = a[2*(j-1)]
            dw1dn_ijc = (aodd - 1) * dw1dn[i, j-1, c]
            dw1dn_ijc += aeven * (dq1dtan[i, j, c] - im * dq2dtan[i, j, c])
            dw1dn_ijc -= aodd * (dq1dtan[i, j-1, c] - im * dq2dtan[i, j-1, c])
            dw1dn_ijc += sig[2*(j-1)] * (q1[i, j, c] - im * q2[i, j, c])
            dw1dn_ijc -= sig[2*(j-1)-1] * (q1[i, j-1, c] - im * q2[i, j-1, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end
    end

    dw2dn[i, Q+1, c] = zero(eltype(dw2dn))

    @inbounds if i <= I
        @unroll for j = Q+1:-1:2
            aodd = a[2*(j-1)-1]
            dw2dn_ijc = (aodd - 1) * dw2dn[i, j, c]
            dw2dn_ijc += a[2*(j-1)] * (dq1dtan[i, j, c] + im * dq2dtan[i, j, c])
            dw2dn_ijc -= aodd * (dq1dtan[i, j-1, c] + im * dq2dtan[i, j-1, c])
            dw2dn_ijc += sig[2*(j-1)] * (q1[i, j, c] + im * q2[i, j, c])
            dw2dn_ijc -= sig[2*(j-1)-1] * (q1[i, j-1, c] + im * q2[i, j-1, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i, j-1, c] = dw2dn_ijc
        end
    end
end

@kernel inbounds=true unsafe_indices=true function crbc_recursion_left!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    m,
    a,
    sig,
    ::Val{J},
    ::Val{Q},
) where {J, Q}
    j = @index(Local, Linear)
    c = @index(Group, Linear)

    if j <= J
        @unroll for itmp = Q:1
            i = Q + 2 - j
            aodd = a[2*(i-1)-1]
            aeven = a[2*(i-1)]

            dw1dn_ijc = (aodd - 1) * dw1dn[i+1, j, c]
            dw1dn_ijc -= aeven * (dq1dtan[i, j, c] - dq2dtan[i, j, c])
            dw1dn_ijc += aodd * (dq1dtan[i+1, j, c] - dq2dtan[i+1, j, c])
            dw1dn_ijc -= sig[2*(i-1)] * (q1[i, j, c] - q2[i, j, c])
            dw1dn_ijc += sig[2*(i-1)-1] * (q1[i+1, j, c] - q2[i+1, j, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end

        dw2dn[1, j, c] = zero(eltype(dw2dn))

        @unroll for i = 1:Q
            pidx = Q + 2 - i
            aodd = a[2*(pidx-1)-1]
            aeven = a[2*(pidx-1)]

            dw2dn_ijc = (aeven - 1) * dw2dn[i, j, c]
            dw2dn_ijc -= a[2*(j-1)] * (dq1dtan[i, j, c] + dq2dtan[i, j, c])
            dw2dn_ijc += aodd * (dq1dtan[i+1, j, c] + dq2dtan[i+1, j, c])
            dw2dn_ijc -= sig[2*(pidx-1)] * (q1[i, j, c] + q2[i, j, c])
            dw2dn_ijc += sig[2*(pidx-1)-1] * (q1[i+1, j, c] + q2[i+1, j, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i+1, j, c] = dw2dn_ijc
        end
    end
end

@kernel inbounds=true unsafe_indices=true function crbc_recursion_right!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    m,
    a,
    sig,
    ::Val{J},
    ::Val{Q},
) where {J, Q}
    j = @index(Local, Linear)
    c = @index(Group, Linear)

    if j <= J
        @unroll for i = 2:Q+1
            aodd = a[2*(i-1)-1]
            aeven = a[2*(i-1)]

            dw1dn_ijc = (aodd - 1) * dw1dn[i-1, j, c]
            dw1dn_ijc += aeven * (dq1dtan[i, j, c] + dq2dtan[i, j, c])
            dw1dn_ijc -= aodd * (dq1dtan[i-1, j, c] + dq2dtan[i-1, j, c])
            dw1dn_ijc += sig[2*(i-1)] * (q1[i, j, c] + q2[i, j, c])
            dw1dn_ijc -= sig[2*(i-1)-1] * (q1[i-1, j, c] + q2[i-1, j, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end

        dw2dn[Q+1, j, c] = zero(eltype(dw2dn))

        @unroll for i = Q+1:-1:2
            aodd = a[2*(i-1)-1]
            aeven = a[2*(i-1)]

            dw2dn_ijc = (aeven - 1) * dw2dn[i, j, c]
            dw2dn_ijc += aeven * (dq1dtan[i, j, c] - dq2dtan[i, j, c])
            dw2dn_ijc -= aodd * (dq1dtan[i-1, j, c] - dq2dtan[i-1, j, c])
            dw2dn_ijc += sig[2*(i-1)] * (q1[i, j, c] - q2[i, j, c])
            dw2dn_ijc -= sig[2*(i-1)-1] * (q1[i-1, j, c] - q2[i-1, j, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i-1, j, c] = dw2dn_ijc
        end
    end
end

function crbc_rhs!(
    dq,
    q,
    dwdn,
    dqdtan,
    grid,
    invwJ,
    DT,
    materialparams,
    a,
    sig,
    cm;
    orient = "top",
)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    if BC == :crbc_tb
        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        block = (last(size(dq)), 1, 1)
        #FIXME: waist of threads!!
        crbc_tangent_tb!(backend, workgroup)(
            dqdtan,
            q,
            dRdX,
            wJ,
            invwJ,
            DT,
            materialparams,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* block,
        )

        J = maximum(size(cell))
        C = max(128 ÷ J, 1)
        workgroup = (J, C)
        block = (1, cld(last(size(dq)),C))
        crbc_surface_tb!(backend, workgroup)(
            dqdtan,
            q,
            fm.vmapM,
            fm.vmapP,
            n,
            wsJ,
            invwJ,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* block,
        )
    elseif BC == :crbc_lr
        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (last(size(dq)), 1, 1)
        crbc_tangent_lr!(backend, workgroup)(
            dqdtan,
            q,
            dRdX,
            wJ,
            invwJ,
            DT,
            materialparams,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
        J = maximum(size(cell))
        C = max(128 ÷ J, 1)
        workgroup = (J, C)
        blocks = (1, cld(last(size(dq)),C))
        crbc_surface_lr!(backend, workgroup)(
            dqdtan,
            q,
            fm.vmapM,
            fm.vmapP,
            n,
            wsJ,
            invwJ,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    end

    if orient == "top"
        @assert BC == :crbc_tb

        I = first(size(cell))
        crbc_recursion_top!(backend, I)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            m,
            a,
            sig,
            Val(size(cell,1)),
            Val(size(cell, 2) - 1);
            ndrange = I*last(size(dq)),
        )

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
        crbc_volume_top!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    elseif orient == "bottom"
        @assert BC == :crbc_tb
        I = first(size(cell))
        crbc_recursion_bottom!(backend, I)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            m,
            a,
            sig,
            Val(size(cell,1)),
            Val(size(cell, 2) - 1);
            ndrange = I*last(size(dq)),
        )

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
        crbc_volume_bottom!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    elseif orient == "left"
        @assert BC == :crbc_lr
        J = size(cell,2)
        crbc_recursion_left!(backend, J)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            m,
            a,
            sig,
            Val(size(cell,2)),
            Val(size(cell, 1) - 1);
            ndrange = last(size(dq)),
        )

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (last(size(dq)), 1, 1)
        crbc_volume_left!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )

    elseif orient == "right"
        @assert BC == :crbc_lr
        J = size(cell,2)
        crbc_recursion_right!(backend, (J, 1))(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            m,
            a,
            sig,
            Val(size(cell,2)),
            Val(size(cell, 1) - 1);
            ndrange = (J, last(size(dq))),
        )

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (last(size(dq)), 1, 1)
        crbc_volume_right!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
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
    workgroup = (size(cell)..., C)
    blocks = (cld(last(size(dq)), C), 1, 1)
    rhs_volume!(backend, workgroup)(
        dq,
        q,
        dRdX,
        wJ,
        invwJ,
        DT,
        materialparams,
        Val(size(cell)),
        Val(C),
        Val(last(size(dq)));
        ndrange = workgroup .* blocks,
    )


    finish!(q, cm)

    J = maximum(size(cell))
    C = max(128 ÷ J, 1)
    # J  x C workgroup sizes to evaluate  multiple elements on one wg.
    workgroup = (J, C)
    blocks = (1, cld(last(size(dq)),C))
    rhs_surface!(backend, workgroup)(
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
        Val(C),
        Val(last(size(dq)));
        ndrange = workgroup .* blocks,
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
    elseif BC == :crbc_lr
        periodicity = (false, true)
    else
        periodicity = (true, false)
    end
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(gm)


    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates)) / 2^L
    dt = cfl * dx / (maximum(N) + 1)^2
    @info "dt = $(dt) w/ dx = $(dx)"

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
                vtkfile["Re(q1)"] = Adapt.adapt(Array, P * real.(first.(q)))
                vtkfile["Im(q1)"] = Adapt.adapt(Array, P * imag.(first.(q)))
                vtkfile["Re(q2)"] = Adapt.adapt(Array, P * real.(last.(q)))
                vtkfile["Im(q2)"] = Adapt.adapt(Array, P * imag.(last.(q)))
                #vtkfile["m"] = Adapt.adapt(Array, matparam)
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

    fn(x, y) = f(x, delta_y)
    fs(x, y) = f(x, -delta_y)
    fe(x, y) = f(delta_x, y)
    fw(x, y) = f(-delta_x, y)

    materialparams_H = zeros(size(wJ))
    gridpoints_H = adapt(Array, grid.points)

    coefmap!(materialparams_H, gridpoints_H, f)
    materialparams = adapt(AT, materialparams_H)

    #jl # initialize state
    q = GridArray{SVector{2,ComplexF64}}(undef, grid)

    backend = Raven.get_backend(q)
    if convergetest == true
        exactwithinterface!(backend, (size(cell)..., 1))(
            q,
            points(grid),
            materialparams,
            0.0,
            Val(size(cell));
            ndrange = size(q),
        )
    else
        gaussian!(backend, (size(cell)..., 1))(
            q,
            points(grid),
            Val(size(cell));
            ndrange = size(q),
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

        if first(gridpoints_H[end, end, j]) ≈ delta_x  #EAST
            bc_H[2, j] = 3
        end

        if first(gridpoints_H[1, 1, j]) ≈ - delta_x  #West
            bc_H[1, j] = 3
        end
    end

    bc = adapt(AT, bc_H)
    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #%%%%%%%%%%%%%#
    #  CRBC DATA  #
    #%%%%%%%%%%%%%#

    matlabdata = MAT.matread("optimal_cosines_data.mat")
    a = SVector(matlabdata["a"]...)
    sig = SVector(matlabdata["sig"]...)

    data = similar(q)
    data .= Ref(zero(eltype(q)))
    if BC == :crbc_lr || BC == :crbc_tb
        Q = length(a) ÷ 2
        crbc_cell_top_bottom = LobattoCell{FT,AT}(N[1] + 1, Q + 1)
        crbc_cell_left_right = LobattoCell{FT,AT}(Q + 1, N[2] + 1)

        data .= Ref(zero(eltype(q)))

        # LEFT
        crbc_coordinates_left = (
            range(FT(-delta_x - 1.0), stop = FT(-delta_x), length = 2),
            range(FT(-delta_y), stop = FT(delta_y), length = K * 2^L + 1),
        )
        crbc_gm_left = GridManager(
            crbc_cell_left_right,
            brick(crbc_coordinates_left, (false, true));
            comm = comm,
            min_level = 0,
        )
        crbc_grid_left = generate(crbc_gm_left)
        crbc_q_left = GridArray{eltype(q)}(undef, crbc_grid_left)
        crbc_dq_left = similar(crbc_q_left)
        crbc_dwdn_left = similar(crbc_q_left)
        crbc_dqdtan_left = similar(crbc_q_left)
        crbc_q_left .= Ref(zero(eltype(q)))
        crbc_dq_left .= Ref(zero(eltype(q)))
        crbc_dwdn_left .= Ref(zero(eltype(q)))
        crbc_dqdtan_left .= Ref(zero(eltype(q)))

        _, _, crbc_wJ_left = components(first(volumemetrics(crbc_grid_left)))
        crbc_invwJ_left = inv.(crbc_wJ_left)
        crbc_DT_left = derivatives_1d(crbc_cell_left_right)

        crbc_materialparams_left_H = zeros(size(crbc_wJ_left))
        crbc_gridpoints_left_H = adapt(Array, crbc_grid_left.points)
        coefmap!(crbc_materialparams_left_H, crbc_gridpoints_left_H, fw)
        crbc_materialparams_left = adapt(AT, crbc_materialparams_left_H)

        #FIXME: This isnt great but you pay the cost once
        crbc_interface_left =
            SVector(findall(≈(-delta_x), adapt(Array, first.(points(crbc_grid_left))))...)
        bulk_interface_left = SVector(findall(≈(-delta_x), adapt(Array, first.(points(grid))))...)
        #FIXME: end

        # RIGHT
        crbc_coordinates_right = (
            range(FT(delta_x), stop = FT(delta_x + 1.0), length = 2),
            range(FT(-delta_y), stop = FT(delta_y), length = K * 2^L + 1),
        )
        crbc_gm_right = GridManager(
            crbc_cell_left_right,
            brick(crbc_coordinates_right, (false, true));
            comm = comm,
            min_level = 0,
        )

        crbc_grid_right = generate(crbc_gm_right)
        crbc_q_right = GridArray{eltype(q)}(undef, crbc_grid_right)
        crbc_q_right .= Ref(zero(eltype(q)))
        crbc_dq_right = similar(crbc_q_right)
        crbc_dwdn_right = similar(crbc_q_right)
        crbc_dqdtan_right = similar(crbc_q_right)
        crbc_dq_right .= Ref(zero(eltype(q)))
        crbc_dwdn_right .= Ref(zero(eltype(q)))
        crbc_dqdtan_right .= Ref(zero(eltype(q)))

        _, _, crbc_wJ_right = components(first(volumemetrics(crbc_grid_right)))
        crbc_invwJ_right = inv.(crbc_wJ_right)
        crbc_DT_right = derivatives_1d(crbc_cell_left_right)

        crbc_materialparams_right_H = zeros(size(crbc_wJ_right))
        crbc_gridpoints_right_H = adapt(Array, crbc_grid_right.points)
        coefmap!(crbc_materialparams_right_H, crbc_gridpoints_right_H, fe)
        crbc_materialparams_right = adapt(AT, crbc_materialparams_right_H)

        #FIXME: This isnt great but you pay the cost once
        crbc_interface_right =
            SVector(findall(≈(delta_x), adapt(Array, first.(points(crbc_grid_right))))...)
        bulk_interface_right = SVector(findall(≈(delta_x), adapt(Array, first.(points(grid))))...)
        #FIXME: end

        # TOP
        crbc_coordinates_top = (
            range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
            range(FT(delta_y), stop = FT(delta_y + 1.0), length = 2),
        )
        crbc_gm_top = GridManager(
            crbc_cell_top_bottom,
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
        crbc_DT_top = derivatives_1d(crbc_cell_top_bottom)

        crbc_materialparams_top_H = zeros(size(crbc_wJ_top))
        crbc_gridpoints_top_H = adapt(Array, crbc_grid_top.points)
        coefmap!(crbc_materialparams_top_H, crbc_gridpoints_top_H, fn)
        crbc_materialparams_top = adapt(AT, crbc_materialparams_top_H)


        #FIXME: This isnt great but you pay the cost once
        crbc_interface_top = SVector(findall(≈(delta_y), adapt(Array, last.(points(crbc_grid_top))))...)
        bulk_interface_top = SVector(findall(≈(delta_y), adapt(Array, last.(points(grid))))...)
        #FIXME: end

        # BOTTOM
        crbc_coordinates_bottom = (
            range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
            range(FT(-delta_y - 1.0), stop = FT(-delta_y), length = 2),
        )
        crbc_gm_bottom = GridManager(
            crbc_cell_top_bottom,
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
        crbc_DT_bottom = derivatives_1d(crbc_cell_top_bottom)

        _, _, crbc_wJ_bottom = components(first(volumemetrics(crbc_grid_bottom)))
        crbc_materialparams_bottom_H = zeros(size(crbc_wJ_bottom))
        crbc_gridpoints_bottom_H = adapt(Array, crbc_grid_bottom.points)
        coefmap!(crbc_materialparams_bottom_H, crbc_gridpoints_bottom_H, fs)
        crbc_materialparams_bottom = adapt(AT, crbc_materialparams_bottom_H)


        #FIXME: This isnt great but you pay the cost once
        crbc_interface_bottom =
            SVector(findall(≈(-delta_y), adapt(Array, last.(points(crbc_grid_bottom))))...)
        bulk_interface_bottom = SVector(findall(≈(-delta_y), adapt(Array, last.(points(grid))))...)
        #FIXME: end
    end # crbc initialization

    #jl # initial output
    step = 0
    time = FT(0)

    do_output(step, time, q, materialparams, grid)
    #do_output(step, time, crbc_dwdn_right, crbc_q_right, crbc_grid_right)
    #do_output(step, time, crbc_dwdn_left, crbc_q_left, crbc_grid_left)

    progress_stepwidth = cld(numberofsteps, progresswidth)
        elapsed = @elapsed begin
        for step = 1:numberofsteps
            elapsed_per_step = @elapsed begin
            if time + dt > timeend
                dt = timeend - time
            end

            for stage in eachindex(RKA, RKB)
                @. dq *= RKA[stage]
                if BC == :crbc_tb || BC == :crbc_lr
                    @. crbc_dq_top *= RKA[stage]
                    @. crbc_dq_bottom *= RKA[stage]
                    @. crbc_dq_left *= RKA[stage]
                    @. crbc_dq_right *= RKA[stage]

                    if BC == :crbc_tb
                            T = 256
                            idxcopyP!(backend, T)(
                                crbc_dwdn_top,
                                crbc_interface_top,
                                q,
                                bulk_interface_top,
                                Complex{FT}.(SA[1 -im; 1 im]);
                                ndrange = size(crbc_interface_top,1)
                            )

                            idxcopyP!(backend, T)(
                                crbc_dwdn_bottom,
                                crbc_interface_bottom,
                                q,
                                bulk_interface_bottom,
                                Complex{FT}.(SA[-1 -im; -1 im]);
                                ndrange = size(crbc_interface_bottom,1)
                            )
                    elseif BC == :crbc_lr
                        T = 256
                        idxcopyP!(backend, T)(
                            crbc_dwdn_left,
                            crbc_interface_left,
                            q,
                            bulk_interface_left,
                            Complex{FT}.(SA[-1 1; -1 -1]);
                            ndrange = size(crbc_interface_left,1)
                        )

                        idxcopyP!(backend, T)(
                            crbc_dwdn_right,
                            crbc_interface_right,
                            q,
                            bulk_interface_right,
                            Complex{FT}.(SA[1 1; 1 -1]);
                            ndrange = size(crbc_interface_right,1)
                        )
                    end

                    if BC == :crbc_tb
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
                    elseif BC == :crbc_lr
                        crbc_rhs!(
                            crbc_dq_left,
                            crbc_q_left,
                            crbc_dwdn_left,
                            crbc_dqdtan_left,
                            crbc_grid_left,
                            crbc_invwJ_left,
                            crbc_DT_left,
                            crbc_materialparams_left,
                            a,
                            sig,
                            cm;
                            orient = "left",
                        )

                        crbc_rhs!(
                            crbc_dq_right,
                            crbc_q_right,
                            crbc_dwdn_right,
                            crbc_dqdtan_right,
                            crbc_grid_right,
                            crbc_invwJ_right,
                            crbc_DT_right,
                            crbc_materialparams_right,
                            a,
                            sig,
                            cm;
                            orient = "right",
                        )
                    end

                    if BC == :crbc_tb
                            idxcopyP!(backend, T)(
                                data,
                                bulk_interface_top,
                                crbc_dwdn_top,
                                crbc_interface_top,
                                Complex{FT}.(SA[0.5 0.5; 0.5*im -0.5*im]);
                                ndrange = size(crbc_interface_top,1)
                            )

                            idxcopyP!(backend, T)(
                                data,
                                bulk_interface_bottom,
                                crbc_dwdn_bottom,
                                crbc_interface_bottom,
                                Complex{FT}.(SA[-0.5 -0.5; 0.5*im -0.5*im]);
                                ndrange = size(crbc_interface_bottom,1)
                            )
                    elseif BC == :crbc_lr
                        idxcopyP!(backend, T)(
                            data,
                            bulk_interface_left,
                            crbc_dwdn_left,
                            crbc_interface_left,
                            Complex{FT}.(SA[-0.5 -0.5; 0.5 -0.5]);
                            ndrange = size(crbc_interface_left,1)
                        )

                        idxcopyP!(backend, T)(
                            data,
                            bulk_interface_right,
                            crbc_dwdn_right,
                            crbc_interface_right,
                            Complex{FT}.(SA[0.5 0.5; 0.5 -0.5]);
                            ndrange = size(crbc_interface_right,1)
                        )
                    end
                end


                if datatype == :synthetic
                    exactwithinterface!(backend, (size(cell)..., 1))(
                        data,
                        points(grid),
                        materialparams,
                        time,
                        Val(size(cell));
                        ndrange = size(q),
                    )
                end

                rhs!(dq, q, grid, data, invwJ, DT, materialparams, bc, cm)

                @. q += RKB[stage] * dt * dq

                if BC == :crbc_tb || BC == :crbc_lr
                    @. crbc_q_top += RKB[stage] * dt * crbc_dq_top
                    @. crbc_q_bottom += RKB[stage] * dt * crbc_dq_bottom
                    @. crbc_q_left += RKB[stage] * dt * crbc_dq_left
                    @. crbc_q_right += RKB[stage] * dt * crbc_dq_right
                end
            end
            time += dt
            #data = Adapt.adapt(AT, data_h)

            do_output(step, time, q, materialparams, grid)
            #do_output(step, time, crbc_dwdn_right, crbc_q_right, crbc_grid_right)
            #do_output(step, time, crbc_dwdn_left, crbc_q_left, crbc_grid_left)

            energy_output(step, energy, q, grid)
        end # elapsed time
        if rank == 0 && progress && mod(step, progress_stepwidth) == 0
            print(
                "\r" *
                raw"-\|/"[cld(step, progress_stepwidth)%4+1] *
                "="^cld(step, progress_stepwidth) *
                " "^(progresswidth - cld(step, progress_stepwidth)) *
                "|",
            )

            timerem = elapsed_per_step*(numberofsteps-step)
            hours = div(timerem, 60*60)
            min = div(timerem, 60)-hours*60
            sec = timerem-hours*60^2-min*60
            @printf "%02i:%02i:%02i (time remaining)" hours min sec
        end
    end # time step for loop
    end #total elapsed time

    if rank == 0
        println(
            "\r" *
            raw"-\|/"[cld(step, progress_stepwidth)%4+1] *
            "="^progresswidth *
            "|",
        )
        hours = div(elapsed, 60*60)
        min = div(elapsed, 60)-hours*60
        sec = elapsed-hours*60^2-min*60
        @printf "Run complete: %02i:%02i:%02i (total time)\n" hours min sec
    end

    do_output(numberofsteps, timeend, q, materialparams, grid)
    #do_output(numberofsteps, timeend, crbc_dwdn_right, crbc_q_right, crbc_grid_right)
    #do_output(numberofsteps, timeend, crbc_dwdn_left, crbc_q_left, crbc_grid_left)
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
            ndrange = size(q),
        )

        err = sqrt(sum(Adapt.adapt(Array, wJ .* abs2.(q .- qexact))))

        rank == 0 && mkpath(vtkdir)
        pvd = rank == 0 ? paraview_collection("conv") : nothing

        do_output = function (L, time, q1, q2, grid)
            if outputvtk
                cell = referencecell(grid)
                cd(vtkdir) do
                    filename = "Level$(lpad(L, 2, '0'))"
                    vtkfile = vtk_grid(filename, grid)
                    P = toequallyspaced(cell)
                    diff = q1 .- q2
                    vtkfile["|e|²"] = Adapt.adapt(Array, P * abs2.(diff))
                    vtkfile["Re(e1)"] = Adapt.adapt(Array, P * real.(first.(diff)))
                    vtkfile["Im(e1)"] = Adapt.adapt(Array, P * imag.(first.(diff)))
                    vtkfile["Re(e2)"] = Adapt.adapt(Array, P * real.(last.(diff)))
                    vtkfile["Im(e2)"] = Adapt.adapt(Array, P * imag.(last.(diff)))
                    vtk_save(vtkfile)
                    if rank == 0
                        pvd[time] = vtkfile
                    end
                end
            end
        end

        do_output(L, 0.0, q, qexact, grid)


        return q, grid, energy, err
    end

    return q, grid, energy, nothing
end

let
    FT = Float64
    #AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array
    AT = Array

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if outputvtk || bigrun
        if CUDA.functional()
            CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
            CUDA.allowscalar(false)
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

        _, _, energy, _ =
            run(FT, AT, N, K, Lout; outputvtk = outputvtk, progress = true, vtkdir, comm)
        if outputvtk
            println("Energy % change:", energy[end]/energy[1])
            cd(vtkdir) do
                open("energy.txt", "w") do file
                    e0 = energy[1]
                    for idx = 1:length(energy)
                        println(file, energy[idx] / e0)
                    end
                end
            end
            rank == 0 && @info "Finished, vtk output written to $vtkdir"
        end
    end # outputvtk || bigrun

    if convergetest
        rank == 0 && @info "Starting convergence study h-refinement"
        err = zeros(FT, numlevels)
        @assert m > 0 "Exact solution requires m > 0"

        for l = 1:numlevels
            L = l - 1
            totalcells = (K * 2^L, K * 2^L)
            dofs = prod(totalcells) * prod(1 .+ N)
            _, _, _, err[l] =
                run(FT, AT, N, K, L; outputvtk = outputvtk, progress = true, vtkdir, comm)

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
                            vtkfile["Re(e1)"] =
                                Adapt.adapt(Array, P * real.(first.(q1 .- q2)))
                            vtkfile["Re(e2)"] =
                                Adapt.adapt(Array, P * real.(last.(q1 .- q2)))

                            vtkfile["Im(e1)"] =
                                Adapt.adapt(Array, P * imag.(first.(q1 .- q2)))
                            vtkfile["Im(e2)"] =
                                Adapt.adapt(Array, P * imag.(last.(q1 .- q2)))

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
