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
# Maybe stray synchronize I removed is causing issues
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
using SparseArrays
using WriteVTK

using MAT
using SpecialFunctions
using AMD
using LinearSolve
using CUDA, CUDA.CUSPARSE

using NVTX
NVTX.enable_gc_hooks(; gc=true, alloc=true, free=true)
NVTX.enable_inference_hook()

#https://github.com/guiyrt/MLinJulia/blob/5b0cf79258848e2723dc270b550066cac07af5a9/src/calodiffusion/profile_cuda.jl#L3
import Base.abs2

const convergetest = false
const empericalconvergetest = false
const outputvtk = true
const outputprogress = true
const bigrun = true
const numlevels = 3
const Lout = 1
const mtemp = -10.0 #, -75.0
const BC = :rbc # options: :rbc :forbc
const flux = :upwind    # options: :upwind :central
const matcoef = :smooth  # options: :constant :disc :smooth

const K = 16
const delta_x = 1.0
const delta_y = 1.0
const pulsewidth = 0.9
const xcenter = 0.0
const ycenter = 0.0
#const f(x, y) = y - ycenter > 0 ? mtemp : -mtemp
#const f(x, y) = x - xcenter > 0 ? mtemp : -mtemp
#const f(x, y) = mtemp*y
f(x,y) = track(x, y)
const timeend = 1.5
const polydegree = 4
const N = (polydegree, polydegree)
const xperiodic = false
const yperiodic = false
const crbcdatafile = "optimal_cosines_data.mat"
const progresswidth = 30

function abs2(v::SVector{2,ComplexF64})
    return abs(real(v' * v))
end

function track(x, y)
    sigma = mtemp
    length = 0.8
    width = 0.5
    s = sqrt(pi)/(2.0*sigma)
    X = max(abs(x) - length/2, 0.0)
    return sigma * erf(s*(sqrt(X^2+y^2)-width/2))
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

@kernel function exactwithoutinterface!(q, mesh, m, t, ::Val{N}) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Group, NTuple)
    x = mesh[i,j,c]
    ab = -2*pi^2
    temp = sqrt(Complex(ab - m^2))
    XY = exp(-im * pi * (x[1] + x[2]))
    f1 = (1 + im) * XY * (-temp * cosh(t*temp)+im*m*sinh(t*temp))/pi
    f2 = 2 * XY * sinh(t*temp)
    qvec = SVector{2,eltype(eltype(q))}(f1, f2)

    q[i, j, c] = qvec
end

@kernel function gaussian!(q, mesh, ::Val{N}) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Group, NTuple)

    x = mesh[i, j, c]
    qvec = SVector{2,eltype(eltype(q))}(1, 1) # vert (1, -im), horz (1, 1)

    #temp = exp(-(mtemp*(x[1] - xcenter))^2) * exp(-(mtemp*(x[2] - ycenter))^2)
    temp = exp(-abs(mtemp)/2*((x[1] - xcenter)^2+(x[2]-ycenter)^2))

    q[i, j, c] = temp * qvec
end


@kernel function bump!(q, mesh, ::Val{N}) where {N}
    i, j, _ = @index(Local, NTuple)
    _, _, c = @index(Group, NTuple)

    x = mesh[i, j, c]
    qvec = SVector{2,eltype(eltype(q))}(1, -im) # vert (1, -im), horz (1, 1)
    a = 0.0
    if -pulsewidth < (x[1] - xcenter) < pulsewidth &&
       -pulsewidth < (x[2] - ycenter) < pulsewidth
        temp =
            exp(1 / (((x[1] - xcenter) / pulsewidth)^2 - 1)) *
            exp(1 / (((x[2] - ycenter) / pulsewidth)^2 - 1))
    else
        temp = 0
    end
    q[i, j, c] = temp * qvec
end

@kernel function setcornerrhs_incomming!(rhs, data, idx, alpha, beta, eqnstart)
    i = @index(Global, Linear)
    rhs[eqnstart+i-1] = alpha * data[idx[i]][1] + beta * data[idx[i]][2]
end

@kernel function setcornerrhs_recursion_j!(rhs, q, a, sig, Q, m, eqnstart)
    jk = @index(Global, Linear)
    k, jt = fldmod1(jk, Q) # k=1:Q+1, j=2:Q+1
    j = jt + 1
    eqn = eqnstart + 2*((j-1) + (k-1)*Q)-2
    aeven = a[2*(j-1)]
    aodd = a[2*(j-1)-1]
    rhs[eqn] = (-aeven * im * m + sig[2*(j-1)]) * q[j, k][1]
    rhs[eqn] += (aodd * im * m - sig[2*(j-1)-1]) * q[j-1, k][1]
    rhs[eqn+1] = (-aodd * im * m - sig[2*(j-1)-1]) * q[j-1, k][2]
    rhs[eqn+1] += (aeven * im * m + sig[2*(j-1)]) * q[j, k][2]
end

@kernel function setcornerrhs_recursion_k!(rhs, q, a, sig, Q, m, eqnstart)
    jk = @index(Global, Linear)
    j, kt = fldmod1(jk, Q) # j=1:Q+1, K=2:Q+1
    k = kt + 1
    eqn = eqnstart + 2*((k-1) + (j-1)*Q)-2
    aeven = a[2*(k-1)]
    aodd = a[2*(k-1)-1]
    rhs[eqn] = (-aeven * im * m + sig[2*(k-1)]) * q[j, k][1]
    rhs[eqn] += (aodd * im * m - sig[2*(k-1)-1]) * q[j, k-1][1]
    rhs[eqn + 1] = (-aodd * im * m - sig[2*(k-1)-1]) * q[j, k-1][2]
    rhs[eqn + 1] += (aeven * im * m + sig[2*(k-1)]) * q[j, k][2]
end

function assembleb!(b_gpu, ynormaldata, xnormaldata, q, idx, param, m, n)
    backend = KernelAbstractions.get_backend(b_gpu)
    ynormalidx, xnormalidx = idx
    a, sig, Q = param
    KernelAbstractions.synchronize(backend)
    # [[inc 1] [inc 2] [recursion 1] [recursion 2] [term 1] [term 2]]
    eqn = 1
    setcornerrhs_incomming!(backend)(b_gpu, xnormaldata, xnormalidx, n[1], 1, eqn; ndrange=Q+1)
    eqn += Q+1
    setcornerrhs_incomming!(backend)(b_gpu, ynormaldata, ynormalidx, n[2], -im, eqn; ndrange=Q+1)
    eqn += Q+1
    setcornerrhs_recursion_j!(backend)(b_gpu, q, a, sig, Q, m, eqn; ndrange=Q*(Q+1))
    eqn += 2*Q*(Q+1)
    setcornerrhs_recursion_k!(backend)(b_gpu, q, a, sig, Q, m, eqn; ndrange=Q*(Q+1))
end

function assembleb_cpu!(bb, data1, data2, qq, idx, param, m, n)
    backend = KernelAbstractions.get_backend(bb)
    idx1, idx2 = idx
    a, sig, Q = param
    # [[inc 1] [inc 2] [recursion 1] [recursion 2] [term 1] [term 2]]
    eqn = 1
    # incomming data (Q+1 equations) x-normal-data
    j = 1
    # lefts-right | dwdx = (dudx + nx*dvdx)^- = (nx * u + v)^+ | x-normal-data
    for k = 1:Q+1
        bb[eqn] = n[1] * first(data2[idx2[k]]) + last(data2[idx2[k]])
        eqn = eqn + 1
    end

    # top-bottom | dwdy = (dudy - ny*i*dvdy)^- = (ny * u - i*v)^+ | y-normal-data
    # incomming data (Q+1 equations) y-normal-data
    k = 1
        for j = 1:Q+1
        bb[eqn] = n[2] * first(data1[idx1[j]]) - im * last(data1[idx1[j]])
        eqn = eqn + 1
    end

    # dir j recursions (2Q(Q+1) equations) | x-normal recursion
    for k = 1:Q+1, j = 2:Q+1
        aeven = a[2*(j-1)]
        aodd = a[2*(j-1)-1]
        bb[eqn] = (-aeven * im * m + sig[2*(j-1)]) * first(qq[j, k])
        bb[eqn] += (aodd * im * m - sig[2*(j-1)-1]) * first(qq[j-1, k])
        eqn = eqn + 1
        bb[eqn] = (-aodd * im * m - sig[2*(j-1)-1]) * last(qq[j-1, k])
        bb[eqn] += (aeven * im * m + sig[2*(j-1)]) * last(qq[j, k])
        eqn = eqn + 1
    end

    # dir k recursions (2Q(Q+1) equations) | y-normal recursion
    for j = 1:Q+1, k = 2:Q+1
        aeven = a[2*(k-1)]
        aodd = a[2*(k-1)-1]
        bb[eqn] = (-aeven * im * m + sig[2*(k-1)]) * first(qq[j, k])
        bb[eqn] += (aodd * im * m - sig[2*(k-1)-1]) * first(qq[j, k-1])
        eqn = eqn + 1
        bb[eqn] = (-aodd * im * m - sig[2*(k-1)-1]) * last(qq[j, k-1])
        bb[eqn] += (aeven * im * m + sig[2*(k-1)]) * last(qq[j, k])
        eqn = eqn + 1
   end
end



function assemblecornersystem!(A, param)
    # dof ordering:
    backend = KernelAbstractions.get_backend(A)
    a, Q, n = param

    # [[inc 1] [inc 2] [recursion 1] [recursion 2] [term 1] [term 2]]
    dudxidx = KernelAbstractions.zeros(backend, Int, (Q + 1, Q + 1))
    dvdxidx = KernelAbstractions.zeros(backend, Int, (Q + 1, Q + 1))
    dudyidx = KernelAbstractions.zeros(backend, Int, (Q + 1, Q + 1))
    dvdyidx = KernelAbstractions.zeros(backend, Int, (Q + 1, Q + 1))

    for k = 1:Q+1, j = 1:Q+1
        dudxidx[j, k] = 1 + 4 * (j - 1) + 4 * (Q + 1) * (k - 1)
        dvdxidx[j, k] = 2 + 4 * (j - 1) + 4 * (Q + 1) * (k - 1)
        dudyidx[j, k] = 3 + 4 * (j - 1) + 4 * (Q + 1) * (k - 1)
        dvdyidx[j, k] = 4 + 4 * (j - 1) + 4 * (Q + 1) * (k - 1)
    end

    eqn = 1

    # lefts-right | dwdx = (dudx + nx*dvdx)^- = (nx * u + v)^+ | x-normal-data
    # incomming k data (Q+1 equations) x-normal-data
    j = 1
    for k = 1:Q+1
        A[eqn, dudxidx[j, k]] = 1
        A[eqn, dvdxidx[j, k]] = n[1]
        eqn = eqn + 1
    end

    # incomming j data (Q+1 equations) y-normal-data
    k = 1
    for j = 1:Q+1
        A[eqn, dudyidx[j, k]] = 1
        A[eqn, dvdyidx[j, k]] = -n[2] * im
        eqn = eqn + 1
    end

    # dir j recursions (2Q(Q+1) equations) | x-normal recursion
    for k = 1:Q+1, j = 2:Q+1
        aodd = a[2*(j-1)-1]
        aeven = a[2*(j-1)]
        A[eqn, dudxidx[j-1, k]] = n[1]
        A[eqn, dvdxidx[j-1, k]] = -aodd
        A[eqn, dvdyidx[j-1, k]] = im * aodd

        A[eqn, dudxidx[j, k]] = n[1]
        A[eqn, dvdxidx[j, k]] = aeven
        A[eqn, dvdyidx[j, k]] = -im * aeven

        eqn = eqn + 1

        A[eqn, dudxidx[j-1, k]] = -aodd
        A[eqn, dvdxidx[j-1, k]] = n[1]
        A[eqn, dudyidx[j-1, k]] = -im * aodd

        A[eqn, dudxidx[j, k]] = aeven
        A[eqn, dvdxidx[j, k]] = n[1]
        A[eqn, dudyidx[j, k]] = im * aeven

        eqn = eqn + 1
    end

    # dir recursions (2Q(Q+1) equations) | y-normal recursion
    for j = 1:Q+1, k = 2:Q+1
        aodd = a[2*(k-1)-1]
        aeven = a[2*(k-1)]
        A[eqn, dvdxidx[j, k-1]] = -aodd
        A[eqn, dudyidx[j, k-1]] = n[2]
        A[eqn, dvdyidx[j, k-1]] = im * aodd

        A[eqn, dvdxidx[j, k]] = aeven
        A[eqn, dudyidx[j, k]] = n[2]
        A[eqn, dvdyidx[j, k]] = -im * aeven

        eqn = eqn + 1

        A[eqn, dudxidx[j, k-1]] = -aodd
        A[eqn, dudyidx[j, k-1]] = -im * aodd
        A[eqn, dvdyidx[j, k-1]] = n[2]

        A[eqn, dudxidx[j, k]] = aeven
        A[eqn, dudyidx[j, k]] = im * aeven
        A[eqn, dvdyidx[j, k]] = n[2]

        eqn = eqn + 1
    end

    j = Q + 1
    # Terminations in k recursion (Q+1 equations) x-normal-data
    for k = 1:Q+1
        A[eqn, dudxidx[j, k]] = 1
        A[eqn, dvdxidx[j, k]] = -n[1]
        eqn = eqn + 1
    end

    k = Q + 1
    # Terminations in j recursion (Q+1 equations) y-normal-data
    for j = 1:Q+1
        A[eqn, dudyidx[j, k]] = 1
        A[eqn, dvdyidx[j, k]] = n[2] * im
        eqn = eqn + 1
    end
end

@kernel function datafromcorner!(qto, qto_idx, qfrom, qfrom_idx, P)
    i = @index(Global, Linear)

    if i <= length(qto_idx)
        qto[qto_idx[i]] =
            P * SVector{4,ComplexF64}(
                qfrom[1+4*(qfrom_idx[i]-1)],
                qfrom[2+4*(qfrom_idx[i]-1)],
                qfrom[3+4*(qfrom_idx[i]-1)],
                qfrom[4+4*(qfrom_idx[i]-1)],
            )
    end
end

@kernel inbounds = true unsafe_indices = true function idxcopyP!(
    qto,
    qto_idx,
    qfrom,
    qfrom_idx,
    P,
    ::Val{C},
    ::Val{L},
) where {C,L}
    il = @index(Local, Linear)
    cg = @index(Group , Linear)
    i = (cg - 1) * C + il

    if i <= L
        qto[qto_idx[i]] = P * qfrom[qfrom_idx[i]]
    end
end

# ∂ₜψ = σ₁∂ₓψ+im(y)σ₃ψ
@kernel inbounds = true unsafe_indices = true function crbc_tangent_tb!(
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
            dqijc_update += wJdRdXijc[1] * lDT1[i, m] * σ₁ * lU[m, j, cl] # -rₓDᵣσ₁u
        end

        dq[i, j, c] = invwJijc * dqijc_update
    end
end

# ∂ₜψ = -σ₂∂ᵥψ -im(y)σ₃ψ
@kernel inbounds = true unsafe_indices = true function crbc_tangent_lr!(
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

        @unroll for n = 0x1:N[2]
            dqijc_update += wJdRdXijc[4] * lDT2[j, n] * σ₂ * lU[i, n, cl] # sᵥDₛσ₂u
        end

        dq[i, j, c] = invwJijc * dqijc_update
    end
end


# ∂ₜψ = dqdtan-σ₂∂ᵥψ
@kernel inbounds = true unsafe_indices = true function crbc_volume_top!(
    dq,
    q,
    dqdtan,
    dwdy,
    matpar,
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
        dq[i, j, c] -=
            dqdtan[i, j, c] +
            σ₂ * P * dwdy[i, j, c] +
            im * matpar[i, j, c] * σ₃ * q[i, j, c]
    end
end

@kernel inbounds = true unsafe_indices = true function crbc_volume_bottom!(
    dq,
    q,
    dqdtan,
    dwdy,
    matpar,
    ::Val{N},
    ::Val{C},
    ::Val{S},
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    P = SA[0.5 0.5; -0.5im 0.5im]
    c = (cg - 1) * C + cl

    if c <= S
        dq[i, j, c] -=
            dqdtan[i, j, c] +
            σ₂ * P * dwdy[i, j, c] +
            im * matpar[i, j, c] * σ₃ * q[i, j, c]
    end
end

@kernel inbounds = true unsafe_indices = true function crbc_volume_left!(
    dq,
    q,
    s2dqdy,
    dwdx,
    matpar,
    ::Val{N},
    ::Val{C},
    ::Val{S},
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg - 1) * C + cl
    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    P = SA[0.5 0.5; -0.5 0.5]

    if c <= S
        dq[i, j, c] -=
            σ₁ * P * dwdx[i, j, c] +
            s2dqdy[i, j, c] +
            im * matpar[i, j, c] * σ₃ * q[i, j, c]
    end
end


@kernel inbounds = true unsafe_indices = true function crbc_volume_right!(
    dq,
    q,
    s2dqdy,
    dwdx,
    matpar,
    ::Val{N},
    ::Val{C},
    ::Val{S},
) where {N,C,S}
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg - 1) * C + cl

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    if c <= S
        P = SA[0.5 0.5; 0.5 -0.5]

        dq[i, j, c] -=
            σ₁ * P * dwdx[i, j, c] +
            s2dqdy[i, j, c] +
            im * matpar[i, j, c] * σ₃ * q[i, j, c]
    end
end



# ∂ₜψ = -σ₁∂ₓψ-σ₂∂ᵥψ-im(y)σ₃ψ
@kernel inbounds = true unsafe_indices = true function rhs_volume!(
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
    i, j, cl = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    cg = @index(Group, Linear) # global cell numbering
    c = (cg - 1) * C + cl

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

@kernel inbounds = true unsafe_indices = true function crbc_surface_tb!(
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
    ::Val{S},
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
        idB = bc[1, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]
        if idB == 3 && !xperiodic
            qP = data[idM]
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
        if idB == 3 && !xperiodic
            qP = data[idM]
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
            dq[i, j, c] -= lqflux[i, j, cl]
        end
    end
end

@kernel inbounds = true unsafe_indices = true function crbc_surface_lr!(
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
    ::Val{S},
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
        idB = bc[3, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 3 && !yperiodic
            qP = data[idM]
        end

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
        idB = bc[4, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 3 && !yperiodic
            qP = data[idM]
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
            dq[i, j, c] -= lqflux[i, j, cl]
        end
    end
end


@kernel inbounds = true unsafe_indices = true function rhs_surface!(
    dq,
    q,
    data_lr,
    data_tb,
    vmapM,
    vmapP,
    bc,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
    ::Val{S},
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

        if idB == 3 && !xperiodic
            if BC == :rbc
                qP = data_lr[idM]
            elseif BC == :forbc
                nx = -1
                FORBC = SA[0.5 nx*0.5; 0.5 -0.5*nx]
                qP = FORBC * qM
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

        if idB == 3 && !xperiodic
            if BC == :rbc
                qP = data_lr[idM]
            elseif BC == :forbc
                nx = 1
                FORBC = SA[0.5 nx*0.5; 0.5 -0.5*nx]
                qP = FORBC * qM
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

        if idB == 3 && !yperiodic
            if BC == :rbc
                qP = data_tb[idM]
            elseif BC == :forbc
                ny = -1
                FORBC = SA[0.5 -0.5*ny*im; 0.5*ny*im 0.5]
                qP = FORBC * qM
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

        if idB == 3 && !yperiodic
            if BC == :rbc
                qP = data_tb[idM]
            elseif BC == :forbc
                ny = 1
                FORBC = SA[0.5 -0.5*ny*im; 0.5*ny*im 0.5]
                qP = FORBC * qM
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

@kernel inbounds = true unsafe_indices = true function crbc_recursion_bottom!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    matpar,
    a,
    sig,
    ::Val{I},
    ::Val{Q},
) where {I,Q}
    i = @index(Local, Linear)
    c = @index(Group, Linear)

    if i <= I
        for j = Q:-1:1
            pidx = Q + 2 - j
            aodd = a[2*(pidx-1)-1]
            aeven = a[2*(pidx-1)]
            dw1dn_ijc = (aodd - 1) * dw1dn[i, j+1, c]

            dw1dn_ijc += aeven * (dq1dtan[i, j, c] + im * dq2dtan[i, j, c])
            dw1dn_ijc -= aodd * (dq1dtan[i, j+1, c] + im * dq2dtan[i, j+1, c])

            dw1dn_ijc += im * matpar[i, j, c] * aeven * (q1[i, j, c] - im * q2[i, j, c])
            dw1dn_ijc -=
                im * matpar[i, j+1, c] * aodd * (q1[i, j+1, c] - im * q2[i, j+1, c])

            dw1dn_ijc -= sig[2*(pidx-1)] * (q1[i, j, c] + im * q2[i, j, c])
            dw1dn_ijc += sig[2*(pidx-1)-1] * (q1[i, j+1, c] + im * q2[i, j+1, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end
    end

    dw2dn[i, 1, c] = zero(eltype(dw2dn))

    if i <= I
        @unroll for j = 1:1:Q
            pidx = Q + 2 - j
            aodd = a[2*(pidx-1)-1]
            aeven = a[2*(pidx-1)]
            sigeven = sig[2*(pidx-1)]
            sigodd = sig[2*(pidx-1)-1]

            dw2dn_ijc = (aodd - 1) * dw2dn[i, j, c]
            dw2dn_ijc += aeven * (dq1dtan[i, j, c] - im * dq2dtan[i, j, c])
            dw2dn_ijc -= aodd * (dq1dtan[i, j+1, c] - im * dq2dtan[i, j+1, c])

            dw2dn_ijc += im * matpar[i, j, c] * aeven * (q1[i, j, c] + im * q2[i, j, c])
            dw2dn_ijc -=
                im * matpar[i, j+1, c] * aodd * (q1[i, j+1, c] + im * q2[i, j+1, c])

            dw2dn_ijc -= sigeven * (q1[i, j, c] - im * q2[i, j, c])
            dw2dn_ijc += sigodd * (q1[i, j+1, c] - im * q2[i, j+1, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i, j+1, c] = dw2dn_ijc
        end
    end
end

@kernel inbounds = true unsafe_indices = true function crbc_recursion_top!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    matpar,
    a,
    sig,
    ::Val{I},
    ::Val{Q},
) where {I,Q}
    i = @index(Local, Linear)
    c = @index(Group, Linear)

    if i <= I
        @unroll for j = 2:Q+1
            aodd = a[2*(j-1)-1]
            aeven = a[2*(j-1)]
            dw1dn_ijc = (aodd - 1) * dw1dn[i, j-1, c]
            dw1dn_ijc -= aeven * (dq1dtan[i, j, c] - im * dq2dtan[i, j, c])
            dw1dn_ijc += aodd * (dq1dtan[i, j-1, c] - im * dq2dtan[i, j-1, c])
            dw1dn_ijc -= im * matpar[i, j, c] * aeven * (q1[i, j, c] + im * q2[i, j, c])
            dw1dn_ijc +=
                im * matpar[i, j-1, c] * aodd * (q1[i, j-1, c] + im * q2[i, j-1, c])

            dw1dn_ijc += sig[2*(j-1)] * (q1[i, j, c] - im * q2[i, j, c])
            dw1dn_ijc -= sig[2*(j-1)-1] * (q1[i, j-1, c] - im * q2[i, j-1, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end
    end

    dw2dn[i, Q+1, c] = zero(eltype(dw2dn))

    if i <= I
        @unroll for j = Q+1:-1:2
            aodd = a[2*(j-1)-1]
            aeven = a[2*(j-1)]
            dw2dn_ijc = (aodd - 1) * dw2dn[i, j, c]
            dw2dn_ijc -= aeven * (dq1dtan[i, j, c] + im * dq2dtan[i, j, c])
            dw2dn_ijc += aodd * (dq1dtan[i, j-1, c] + im * dq2dtan[i, j-1, c])
            dw2dn_ijc -= im * matpar[i, j, c] * aeven * (q1[i, j, c] - im * q2[i, j, c])
            dw2dn_ijc +=
                im * matpar[i, j-1, c] * aodd * (q1[i, j-1, c] - im * q2[i, j-1, c])

            dw2dn_ijc += sig[2*(j-1)] * (q1[i, j, c] + im * q2[i, j, c])
            dw2dn_ijc -= sig[2*(j-1)-1] * (q1[i, j-1, c] + im * q2[i, j-1, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i, j-1, c] = dw2dn_ijc
        end
    end
end


@kernel inbounds = true unsafe_indices = true function crbc_recursion_left!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    matpar,
    a,
    sig,
    ::Val{J},
    ::Val{Q},
) where {J,Q}
    j = @index(Local, Linear)
    c = @index(Group, Linear)

    if j <= J
        @unroll for i = Q:-1:1
            pidx = Q + 2 - i
            aodd = a[2*(pidx-1)-1]
            aeven = a[2*(pidx-1)]

            dw1dn_ijc = (aodd - 1) * dw1dn[i+1, j, c]
            dw1dn_ijc += aeven * (dq1dtan[i, j, c] - dq2dtan[i, j, c])
            dw1dn_ijc -= aodd * (dq1dtan[i+1, j, c] - dq2dtan[i+1, j, c])

            dw1dn_ijc += im * matpar[i, j, c] * aeven * (q1[i, j, c] + q2[i, j, c])
            dw1dn_ijc -= im * matpar[i+1, j, c] * aodd * (q1[i+1, j, c] + q2[i+1, j, c])

            dw1dn_ijc -= sig[2*(pidx-1)] * (q1[i, j, c] - q2[i, j, c])
            dw1dn_ijc += sig[2*(pidx-1)-1] * (q1[i+1, j, c] - q2[i+1, j, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end
    end

    dw2dn[1, j, c] = zero(eltype(dw2dn))

    if j <= J
        @unroll for i = 1:Q
            pidx = Q + 2 - i
            aodd = a[2*(pidx-1)-1]
            aeven = a[2*(pidx-1)]

            dw2dn_ijc = (aeven - 1) * dw2dn[i, j, c]
            dw2dn_ijc += aeven * (dq1dtan[i, j, c] + dq2dtan[i, j, c])
            dw2dn_ijc -= aodd * (dq1dtan[i+1, j, c] + dq2dtan[i+1, j, c])

            dw2dn_ijc += im * matpar[i, j, c] * aeven * (q1[i, j, c] - q2[i, j, c])
            dw2dn_ijc -= im * matpar[i+1, j, c] * aodd * (q1[i+1, j, c] - q2[i+1, j, c])

            dw2dn_ijc -= sig[2*(pidx-1)] * (q1[i, j, c] + q2[i, j, c])
            dw2dn_ijc += sig[2*(pidx-1)-1] * (q1[i+1, j, c] + q2[i+1, j, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i+1, j, c] = dw2dn_ijc
        end
    end
end

@kernel inbounds = true unsafe_indices = true function crbc_recursion_right!(
    dw1dn,
    dw2dn,
    dq1dtan,
    dq2dtan,
    q1,
    q2,
    matpar,
    a,
    sig,
    ::Val{J},
    ::Val{Q},
) where {J,Q}
    j = @index(Local, Linear)
    c = @index(Group, Linear)

    if j <= J
        @unroll for i = 2:Q+1
            aodd = a[2*(i-1)-1]
            aeven = a[2*(i-1)]

            dw1dn_ijc = (aodd - 1) * dw1dn[i-1, j, c]
            dw1dn_ijc -= aeven * (dq1dtan[i, j, c] + dq2dtan[i, j, c])
            dw1dn_ijc += aodd * (dq1dtan[i-1, j, c] + dq2dtan[i-1, j, c])
            dw1dn_ijc -= im * matpar[i, j, c] * aeven * (q1[i, j, c] - q2[i, j, c])
            dw1dn_ijc += im * matpar[i-1, j, c] * aodd * (q1[i-1, j, c] - q2[i-1, j, c])

            dw1dn_ijc += sig[2*(i-1)] * (q1[i, j, c] + q2[i, j, c])
            dw1dn_ijc -= sig[2*(i-1)-1] * (q1[i-1, j, c] + q2[i-1, j, c])
            dw1dn_ijc /= (aeven + 1)

            dw1dn[i, j, c] = dw1dn_ijc
        end
    end

    dw2dn[Q+1, j, c] = zero(eltype(dw2dn))

    if j <= J
        @unroll for i = Q+1:-1:2
            aodd = a[2*(i-1)-1]
            aeven = a[2*(i-1)]

            dw2dn_ijc = (aeven - 1) * dw2dn[i, j, c]
            dw2dn_ijc -= aeven * (dq1dtan[i, j, c] - dq2dtan[i, j, c])
            dw2dn_ijc += aodd * (dq1dtan[i-1, j, c] - dq2dtan[i-1, j, c])
            dw2dn_ijc -= im * matpar[i, j, c] * aeven * (q1[i, j, c] + q2[i, j, c])
            dw2dn_ijc += im * matpar[i-1, j, c] * aodd * (q1[i-1, j, c] + q2[i-1, j, c])

            dw2dn_ijc += sig[2*(i-1)] * (q1[i, j, c] - q2[i, j, c])
            dw2dn_ijc -= sig[2*(i-1)-1] * (q1[i-1, j, c] - q2[i-1, j, c])
            dw2dn_ijc /= (aodd + 1)

            dw2dn[i-1, j, c] = dw2dn_ijc
        end
    end
end

function corner_copy_data!(out_data1, out_data2, sol, to_idx, Q, n)
    idx1, idx2 = to_idx
    backend = KernelAbstractions.get_backend(sol)

    # lefts-right | dwdx = (dudx + nx*dvdx)^- = (u + nx * v)^+ | x-normal-data
    datafromcorner!(backend)(
        out_data2,
        idx2,
        sol,
        SVector(collect(1:Q+1:(Q+1)^2)...),
        Complex{Float64}.(SA[n[1] 0 0 0; 0 n[1] 0 0]);
        ndrange = Q + 1,
    )

    # top-bottom | dwdy = (dudy - ny*i*dvdy)^- = (u - ny* i*v)^+ | y-normal-data
    datafromcorner!(backend)(
        out_data1,
        idx1,
        sol,
        SVector(collect(1:Q+1)...),
        Complex{Float64}.(SA[0 0 n[2] 0; 0 0 0 n[2]]);
        ndrange = Q + 1,
    )
end

@kernel function crbc_corner_rhs!(dq, q, sol, m, ::Val{Q}) where {Q}
    idx = @index(Global, Linear)
    solidx = 4 * (idx - 1) + 1
    temp = SVector{2,ComplexF64}(
        -sol[solidx+1] + im * sol[solidx+3] - im * m * first(q[idx]),
        -sol[solidx] - im * sol[solidx+2] + im * m * last(q[idx]),
    )
    dq[idx] += temp
end

function crbc_face_rhs!(
    dq,
    q,
    data,
    dwdn,
    dqdtan,
    grid,
    invwJ,
    DT,
    materialparams,
    a,
    sig,
    bc,
    cm,
    orient,
)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    KernelAbstractions.synchronize(backend)
    if orient == "top"
        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
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
            ndrange = workgroup .* blocks,
        )
        KernelAbstractions.synchronize(backend)

        J = maximum(size(cell))
        C = max(128 ÷ J, 1)
        workgroup = (J, C)
        blocks = (1, cld(last(size(dq)), C))
        crbc_surface_tb!(backend, workgroup)(
            dqdtan,
            q,
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
        I = first(size(cell))
        crbc_recursion_top!(backend, I)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            materialparams,
            a,
            sig,
            Val(size(cell, 1)),
            Val(size(cell, 2) - 1);
            ndrange = I * last(size(dq)),
        )
        KernelAbstractions.synchronize(backend)

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
        crbc_volume_top!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            materialparams,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    elseif orient == "bottom"
        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
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
            ndrange = workgroup .* blocks,
        )
        KernelAbstractions.synchronize(backend)

        J = maximum(size(cell))
        C = max(128 ÷ J, 1)
        workgroup = (J, C)
        blocks = (1, cld(last(size(dq)), C))
        crbc_surface_tb!(backend, workgroup)(
            dqdtan,
            q,
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
        I = first(size(cell))
        crbc_recursion_bottom!(backend, I)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            materialparams,
            a,
            sig,
            Val(size(cell, 1)),
            Val(size(cell, 2) - 1);
            ndrange = I * last(size(dq)),
        )
        KernelAbstractions.synchronize(backend)

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
        crbc_volume_bottom!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            materialparams,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    elseif orient == "left"
        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
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
        KernelAbstractions.synchronize(backend)

        J = maximum(size(cell))
        C = max(128 ÷ J, 1)
        workgroup = (J, C)
        blocks = (1, cld(last(size(dq)), C))
        crbc_surface_lr!(backend, workgroup)(
            dqdtan,
            q,
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
        J = size(cell, 2)
        crbc_recursion_left!(backend, J)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            materialparams,
            a,
            sig,
            Val(size(cell, 2)),
            Val(size(cell, 1) - 1);
            ndrange = J * last(size(dq)),
        )
        KernelAbstractions.synchronize(backend)

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
        crbc_volume_left!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            materialparams,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    elseif orient == "right"
        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
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
        KernelAbstractions.synchronize(backend)

        J = maximum(size(cell))
        C = max(128 ÷ J, 1)
        workgroup = (J, C)
        blocks = (1, cld(last(size(dq)), C))
        crbc_surface_lr!(backend, workgroup)(
            dqdtan,
            q,
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
        J = size(cell, 2)
        crbc_recursion_right!(backend, J)(
            components(dwdn)[1],
            components(dwdn)[2],
            components(dqdtan)[1],
            components(dqdtan)[2],
            components(q)[1],
            components(q)[2],
            materialparams,
            a,
            sig,
            Val(size(cell, 2)),
            Val(size(cell, 1) - 1);
            ndrange = J * last(size(dq)),
        )
        KernelAbstractions.synchronize(backend)

        C = max(512 ÷ prod(size(cell)), 1)
        workgroup = (size(cell)..., C)
        blocks = (cld(last(size(dq)), C), 1, 1)
        crbc_volume_right!(backend, workgroup)(
            dq,
            q,
            dqdtan,
            dwdn,
            materialparams,
            Val(size(cell)),
            Val(C),
            Val(last(size(dq)));
            ndrange = workgroup .* blocks,
        )
    end
end


function rhs!(dq, q, grid, data_lr, data_tb, invwJ, DT, materialparams, bc, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)
    start!(q, cm)

    C = max(512 ÷ prod(size(cell)), 1)
    workgroup = (size(cell)..., C)
    blocks = (1, 1, cld(last(size(dq)), C))
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
    blocks = (1, cld(last(size(dq)), C))
    rhs_surface!(backend, workgroup)(
        dq,
        viewwithghosts(q),
        data_lr,
        data_tb,
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
    L,
    crbcparamfile;
    outputvtk = false,
    progress = true,
    vtkdir = "output",
    comm = MPI.COMM_WORLD,
)
    @info "Starting run setup"
    rank = MPI.Comm_rank(comm)
    cell = LobattoCell{FT,AT}((N .+ 1)...)
    coordinates = (
        range(FT(-delta_x), stop = FT(delta_x), length = K + 1),
        range(FT(-delta_y), stop = FT(delta_y), length = K + 1),
    )

    gm = GridManager(
        cell,
        brick(coordinates, (xperiodic, yperiodic));
        comm = comm,
        min_level = L,
    )
    grid = generate(gm)


    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates)) / 2^(numlevels)
    dt = cfl * dx / (maximum(N) + 1)^2
    @info "dt = $(dt) w/ dx = $(dx)"

    numberofsteps = ceil(Int, timeend / dt)
    energy = []
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

    vtk_output = function (step, time, q, matparam, grid)
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
                vtkfile["m"] = Adapt.adapt(Array, matparam)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    cli_status_output = function (step, energy, numberofsteps, elapsed)
        if rank == 0 && progress && mod(step, progress_stepwidth) == 0 && step < numberofsteps
            print(
                "\r" *
                raw"-\|/"[cld(step, progress_stepwidth)%4+1] *
                "="^cld(step, progress_stepwidth) *
                " "^(progresswidth - cld(step, progress_stepwidth)) *
                "|",
            )

            timerem = elapsed * (numberofsteps - step)
            hours = div(timerem, 60 * 60)
            min = div(timerem, 60) - hours * 60
            sec = timerem - hours * 60^2 - min * 60
            if length(energy) > 0
                @printf "%02i:%02i:%02i (time remaining) | %.2f (energy) | " hours min sec energy[end] / energy[1]
            else
                @printf "%02i:%02i:%02i (time remaining) | " hours min sec
            end
        elseif rank == 0 && progress && step == numberofsteps
            println(
                "\r" *
                raw"-\|/"[cld(step, progress_stepwidth)%4+1] *
                "="^progresswidth *
                "|",
            )
            hours = div(elapsed, 60 * 60)
            min = div(elapsed, 60) - hours * 60
            sec = elapsed - hours * 60^2 - min * 60
            @printf "Run complete: %02i:%02i:%02i (total time) |  \n" hours min sec
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
    bump!(backend, (size(cell)..., 1))(q, points(grid), Val(size(cell)); ndrange = size(q))

    #gaussian!(backend, (size(cell)..., 1))(q, points(grid), Val(size(cell)); ndrange = size(q))
    #=
    exactwithinterface!(backend, (size(cell)..., 1))(
        q,
        points(grid),
        materialparams,
        0.0,
        Val(size(cell));
        ndrange = size(q),
    )
    =#

    #jl # storage for RHS
    dq = KernelAbstractions.zeros(backend, eltype(q), size(q))

    # adjust boundary code
    bc_H = adapt(Array, boundarycodes(grid))
    for j = 1:size(bc_H, 2)
        if last(gridpoints_H[end, end, j]) ≈ delta_y  #North
            bc_H[4, j] = 3
        end

        if last(gridpoints_H[1, 1, j]) ≈ -delta_y # South
            bc_H[3, j] = 3
        end

        if first(gridpoints_H[end, end, j]) ≈ delta_x  #EAST
            bc_H[2, j] = 3
        end

        if first(gridpoints_H[1, 1, j]) ≈ -delta_x  #West
            bc_H[1, j] = 3
        end
    end

    bc = adapt(AT, bc_H)
    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #%%%%%%%%%%%%%%%%%%#
    #  CRBC FACE DATA  #
    #%%%%%%%%%%%%%%%%%%#

    matlabdata = MAT.matread(crbcparamfile)
    a = SVector(matlabdata["a"]...)
    sig = SVector(matlabdata["sig"]...)

    data_lr = copy(q)
    data_tb = copy(q)
    if BC == :rbc
        Q = length(a) ÷ 2
        crbc_cell_top_bottom = LobattoCell{FT,AT}(N[1] + 1, Q + 1)
        crbc_cell_left_right = LobattoCell{FT,AT}(Q + 1, N[2] + 1)

        # LEFT
        crbc_coordinates_left = (
            range(FT(-delta_x - 1.0), stop = FT(-delta_x), length = 2),
            range(FT(-delta_y), stop = FT(delta_y), length = K * 2^L + 1),
        )
        crbc_gm_left = GridManager(
            crbc_cell_left_right,
            brick(crbc_coordinates_left, (false, yperiodic));
            comm = comm,
            min_level = 0,
        )
        crbc_grid_left = generate(crbc_gm_left)
        crbc_q_left = GridArray{eltype(q)}(undef, crbc_grid_left)
        crbc_q_left .= Ref(zero(eltype(q)))
        crbc_dq_left = copy(crbc_q_left)
        crbc_dwdn_left = copy(crbc_q_left)
        crbc_dqdtan_left = copy(crbc_q_left)

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
        bulk_interface_left =
            SVector(findall(≈(-delta_x), adapt(Array, first.(points(grid))))...)
        #FIXME: end

        # RIGHT
        crbc_coordinates_right = (
            range(FT(delta_x), stop = FT(delta_x + 1.0), length = 2),
            range(FT(-delta_y), stop = FT(delta_y), length = K * 2^L + 1),
        )
        crbc_gm_right = GridManager(
            crbc_cell_left_right,
            brick(crbc_coordinates_right, (false, yperiodic));
            comm = comm,
            min_level = 0,
        )

        crbc_grid_right = generate(crbc_gm_right)
        crbc_q_right = GridArray{eltype(q)}(undef, crbc_grid_right)
        crbc_q_right .= Ref(zero(eltype(q)))
        crbc_dq_right = copy(crbc_q_right)
        crbc_dwdn_right = copy(crbc_q_right)
        crbc_dqdtan_right = copy(crbc_q_right)

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
        bulk_interface_right =
            SVector(findall(≈(delta_x), adapt(Array, first.(points(grid))))...)
        #FIXME: end

        # TOP
        crbc_coordinates_top = (
            range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
            range(FT(delta_y), stop = FT(delta_y + 1.0), length = 2),
        )
        crbc_gm_top = GridManager(
            crbc_cell_top_bottom,
            brick(crbc_coordinates_top, (xperiodic, false));
            comm = comm,
            min_level = 0,
        )
        crbc_grid_top = generate(crbc_gm_top)
        crbc_q_top = GridArray{eltype(q)}(undef, crbc_grid_top)
        crbc_q_top .= Ref(zero(eltype(q)))
        crbc_dq_top = copy(crbc_q_top)
        crbc_dwdn_top = copy(crbc_q_top)
        crbc_dqdtan_top = copy(crbc_q_top)
        _, _, crbc_wJ_top = components(first(volumemetrics(crbc_grid_top)))
        crbc_invwJ_top = inv.(crbc_wJ_top)
        crbc_DT_top = derivatives_1d(crbc_cell_top_bottom)

        crbc_materialparams_top_H = zeros(size(crbc_wJ_top))
        crbc_gridpoints_top_H = adapt(Array, crbc_grid_top.points)
        coefmap!(crbc_materialparams_top_H, crbc_gridpoints_top_H, fn)
        crbc_materialparams_top = adapt(AT, crbc_materialparams_top_H)


        #FIXME: This isnt great but you pay the cost once
        crbc_interface_top =
            SVector(findall(≈(delta_y), adapt(Array, last.(points(crbc_grid_top))))...)
        bulk_interface_top =
            SVector(findall(≈(delta_y), adapt(Array, last.(points(grid))))...)
        #FIXME: end

        # BOTTOM
        crbc_coordinates_bottom = (
            range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
            range(FT(-delta_y - 1.0), stop = FT(-delta_y), length = 2),
        )
        crbc_gm_bottom = GridManager(
            crbc_cell_top_bottom,
            brick(crbc_coordinates_bottom, (xperiodic, false));
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

        crbc_materialparams_bottom_H = zeros(size(crbc_wJ_bottom))
        crbc_gridpoints_bottom_H = adapt(Array, crbc_grid_bottom.points)
        coefmap!(crbc_materialparams_bottom_H, crbc_gridpoints_bottom_H, fs)
        crbc_materialparams_bottom = adapt(AT, crbc_materialparams_bottom_H)

        #FIXME: This isnt great but you pay the cost once
        crbc_interface_bottom =
            SVector(findall(≈(-delta_y), adapt(Array, last.(points(crbc_grid_bottom))))...)
        bulk_interface_bottom =
            SVector(findall(≈(-delta_y), adapt(Array, last.(points(grid))))...)
        #FIXME: end

        bc_top_H = adapt(Array, boundarycodes(crbc_grid_top))
        gridpoints_H = adapt(Array, crbc_grid_top.points)
        for j = 1:size(bc_top_H, 2)
            if first(gridpoints_H[end, end, j]) ≈ delta_x  #EAST
                bc_top_H[2, j] = 3
            end

            if first(gridpoints_H[1, 1, j]) ≈ -delta_x  #West
                bc_top_H[1, j] = 3
            end
        end

        bc_top = adapt(AT, bc_top_H)

        bc_bottom_H = adapt(Array, boundarycodes(crbc_grid_bottom))
        gridpoints_H = adapt(Array, crbc_grid_bottom.points)
        for j = 1:size(bc_bottom_H, 2)
            if first(gridpoints_H[end, end, j]) ≈ delta_x  #EAST
                bc_bottom_H[2, j] = 3
            end

            if first(gridpoints_H[1, 1, j]) ≈ -delta_x  #West
                bc_bottom_H[1, j] = 3
            end
        end

        bc_bottom = adapt(AT, bc_bottom_H)

        bc_right_H = adapt(Array, boundarycodes(crbc_grid_right))
        gridpoints_H = adapt(Array, crbc_grid_right.points)
        for j = 1:size(bc_right_H, 2)
            if last(gridpoints_H[end, end, j]) ≈ delta_y  #North
                bc_right_H[4, j] = 3
            end

            if last(gridpoints_H[1, 1, j]) ≈ -delta_y # South
                bc_right_H[3, j] = 3
            end
        end

        bc_right = adapt(AT, bc_right_H)

        bc_left_H = adapt(Array, boundarycodes(crbc_grid_left))
        gridpoints_H = adapt(Array, crbc_grid_left.points)
        for j = 1:size(bc_left_H, 2)
            if last(gridpoints_H[end, end, j]) ≈ delta_y  #North
                bc_left_H[4, j] = 3
            end

            if last(gridpoints_H[1, 1, j]) ≈ -delta_y # South
                bc_left_H[3, j] = 3
            end
        end

        bc_left = adapt(AT, bc_left_H)


        #%%%%%%%%%%%%%%%%%%%%#
        #  CRBC CORNER DATA  #
        #%%%%%%%%%%%%%%%%%%%%#
        # 1:bottom/left, 2:bottom/right 3:top/left 4:topright

        crbc_q_corner1 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))
        crbc_q_corner2 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))
        crbc_q_corner3 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))
        crbc_q_corner4 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))

        crbc_dq_corner1 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))
        crbc_dq_corner2 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))
        crbc_dq_corner3 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))
        crbc_dq_corner4 = KernelAbstractions.zeros(backend, eltype(q), (Q + 1, Q + 1))

        crbc_data_top = copy(crbc_q_top)
        crbc_data_right = copy(crbc_q_right)
        crbc_data_left = copy(crbc_q_left)
        crbc_data_bottom = copy(crbc_q_bottom)

        crbc_top_left =
            SVector(findall(≈(-delta_x), adapt(Array, first.(points(crbc_grid_top))))...)
        crbc_top_right =
            SVector(findall(≈(delta_x), adapt(Array, first.(points(crbc_grid_top))))...)
        crbc_bottom_left = SVector(
            findall(≈(-delta_x), adapt(Array, first.(points(crbc_grid_bottom))))[end:-1:1]...,
        )
        crbc_bottom_right = SVector(
            findall(≈(delta_x), adapt(Array, first.(points(crbc_grid_bottom))))[end:-1:1]...,
        )
        crbc_left_top = SVector(
            findall(≈(delta_y), adapt(Array, last.(points(crbc_grid_left))))[end:-1:1]...,
        )
        crbc_left_bottom = SVector(
            findall(≈(-delta_y), adapt(Array, last.(points(crbc_grid_left))))[end:-1:1]...,
        )
        crbc_right_top =
            SVector(findall(≈(delta_y), adapt(Array, last.(points(crbc_grid_right))))...)
        crbc_right_bottom =
            SVector(findall(≈(-delta_y), adapt(Array, last.(points(crbc_grid_right))))...)

        from_idx1 = (crbc_left_bottom, crbc_bottom_left)
        from_idx2 = (crbc_right_bottom, crbc_bottom_right)
        from_idx3 = (crbc_left_top, crbc_top_left)
        from_idx4 = (crbc_right_top, crbc_top_right)

        crbc_x_SW = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)
        crbc_x_SE = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)
        crbc_x_NE = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)
        crbc_x_NW = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)

        crbc_b_SW = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)
        crbc_b_SE = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)
        crbc_b_NW = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)
        crbc_b_NE = KernelAbstractions.zeros(backend, eltype(eltype(q)), 4 * (Q + 1)^2)

        b1 = KernelAbstractions.zeros(CPU(), eltype(eltype(q)), 4 * (Q + 1)^2)
        b2 = KernelAbstractions.zeros(CPU(), eltype(eltype(q)), 4 * (Q + 1)^2)
        b3 = KernelAbstractions.zeros(CPU(), eltype(eltype(q)), 4 * (Q + 1)^2)
        b4 = KernelAbstractions.zeros(CPU(), eltype(eltype(q)), 4 * (Q + 1)^2)

    end # crbc initialization

    #jl # initial output
    step = 0
    time = FT(0)

    vtk_output(step, time, q, materialparams, grid)
    if BC == :rbc
        ASW = KernelAbstractions.zeros(CPU(), Complex{FT}, (4 * (Q + 1)^2, 4 * (Q + 1)^2))
        ASE = KernelAbstractions.zeros(CPU(), Complex{FT}, (4 * (Q + 1)^2, 4 * (Q + 1)^2))
        ANW = KernelAbstractions.zeros(CPU(), Complex{FT}, (4 * (Q + 1)^2, 4 * (Q + 1)^2))
        ANE = KernelAbstractions.zeros(CPU(), Complex{FT}, (4 * (Q + 1)^2, 4 * (Q + 1)^2))
        assemblecornersystem!(ASW, (a, Q, (-1, -1)))
        assemblecornersystem!(ASE, (a, Q, (1, -1)))
        assemblecornersystem!(ANW, (a, Q, (-1, 1)))
        assemblecornersystem!(ANE, (a, Q, (1, 1)))

        d_ASW = CuSparseMatrixCSR(sparse(ASW))
        d_ASE = CuSparseMatrixCSR(sparse(ASE))
        d_ANW = CuSparseMatrixCSR(sparse(ANW))
        d_ANE = CuSparseMatrixCSR(sparse(ANE))

        systemparam = (a, sig, Q)
    end

    @info "Finished setup"

    progress_stepwidth = cld(numberofsteps, progresswidth)
    elapsed = @elapsed begin
        for step = 1:numberofsteps
            elapsed_per_step = @elapsed begin
                if time + dt > timeend
                    dt = timeend - time
                end

                NVTX.@range "RK stage" begin
                for stage in eachindex(RKA, RKB)
                    @. dq *= RKA[stage]
                    if BC == :rbc
                        @. crbc_dq_top *= RKA[stage]
                        @. crbc_dq_bottom *= RKA[stage]
                        @. crbc_dq_left *= RKA[stage]
                        @. crbc_dq_right *= RKA[stage]

                        if !xperiodic && !yperiodic
                            @. crbc_dq_corner1 *= RKA[stage]
                            @. crbc_dq_corner2 *= RKA[stage]
                            @. crbc_dq_corner3 *= RKA[stage]
                            @. crbc_dq_corner4 *= RKA[stage]

                            NVTX.@range "corner rhs assembly" begin
                            assembleb!(
                                crbc_b_SW,
                                crbc_q_left,
                                crbc_q_bottom,
                                crbc_q_corner1,
                                from_idx1,
                                systemparam,
                                f(-delta_x, -delta_y),
                                (-1, -1),
                            )

                            assembleb!(
                                crbc_b_SE,
                                crbc_q_right,
                                crbc_q_bottom,
                                crbc_q_corner2,
                                from_idx2,
                                systemparam,
                                f(delta_x, -delta_y),
                                (1, -1),
                            )

                            assembleb!(
                                crbc_b_NW,
                                crbc_q_left,
                                crbc_q_top,
                                crbc_q_corner3,
                                from_idx3,
                                systemparam,
                                f(-delta_x, delta_y),
                                (-1, 1),
                            )

                            assembleb!(
                                crbc_b_NE,
                                crbc_q_right,
                                crbc_q_top,
                                crbc_q_corner4,
                                from_idx4,
                                systemparam,
                                f(delta_x, delta_y),
                                (1, 1),
                            )
                            end

                            KernelAbstractions.synchronize(backend)

                            #FIXME: !!! SOLVE ON THE GPU
                            NVTX.@range "corner: solve and mem transfer" begin
                                b1 .= Array(crbc_b_SW)
                                b2 .= Array(crbc_b_SE)
                                b3 .= Array(crbc_b_NW)
                                b4 .= Array(crbc_b_NE)

                                #crbc_x_SW .= Array(sparse(ASW) \ b1)
                                #crbc_x_SE .= Array(sparse(ASE) \ b2)
                                #crbc_x_NW .= Array(sparse(ANW) \ b3)
                                #crbc_x_NE .= Array(sparse(ANE) \ b4)
                                crbc_x_SW .= CuArray(sparse(ASW) \ b1)
                                crbc_x_SE .= CuArray(sparse(ASE) \ b2)
                                crbc_x_NW .= CuArray(sparse(ANW) \ b3)
                                crbc_x_NE .= CuArray(sparse(ANE) \ b4)
                            end
                            #FIXME: end !!! SOLVE ON THE GPU

                            KernelAbstractions.synchronize(backend)

                            NVTX.@range "corner rhs" begin
                            crbc_corner_rhs!(backend)(
                                crbc_dq_corner1,
                                crbc_q_corner1,
                                crbc_x_SW,
                                f(-delta_x, -delta_y),
                                Val(Q);
                                ndrange = (Q + 1)^2,
                            )

                            crbc_corner_rhs!(backend)(
                                crbc_dq_corner2,
                                crbc_q_corner2,
                                crbc_x_SE,
                                f(delta_x, -delta_y),
                                Val(Q);
                                ndrange = (Q + 1)^2,
                            )

                            crbc_corner_rhs!(backend)(
                                crbc_dq_corner3,
                                crbc_q_corner3,
                                crbc_x_NW,
                                f(-delta_x, delta_y),
                                Val(Q);
                                ndrange = (Q + 1)^2,
                            )

                            crbc_corner_rhs!(backend)(
                                crbc_dq_corner4,
                                crbc_q_corner4,
                                crbc_x_NE,
                                f(delta_x, delta_y),
                                Val(Q);
                                ndrange = (Q + 1)^2,
                            )
                            end

                            corner_copy_data!(
                                crbc_data_left,
                                crbc_data_bottom,
                                crbc_x_SW,
                                from_idx1,
                                Q,
                                (-1, -1),
                            )

                            corner_copy_data!(
                                crbc_data_right,
                                crbc_data_bottom,
                                crbc_x_SE,
                                from_idx2,
                                Q,
                                (1, -1),
                            )

                            corner_copy_data!(
                                crbc_data_left,
                                crbc_data_top,
                                crbc_x_NW,
                                from_idx3,
                                Q,
                                (-1, 1),
                            )

                            corner_copy_data!(
                                crbc_data_right,
                                crbc_data_top,
                                crbc_x_NE,
                                from_idx4,
                                Q,
                                (1, 1),
                            )
                        end

                        L = size(crbc_interface_top,1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            crbc_dwdn_top,
                            crbc_interface_top,
                            q,
                            bulk_interface_top,
                            Complex{FT}.(SA[1 -im; 1 im]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        L = size(crbc_interface_bottom,1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            crbc_dwdn_bottom,
                            crbc_interface_bottom,
                            q,
                            bulk_interface_bottom,
                            Complex{FT}.(SA[-1 -im; -1 im]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        L = size(crbc_interface_left,1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            crbc_dwdn_left,
                            crbc_interface_left,
                            q,
                            bulk_interface_left,
                            Complex{FT}.(SA[-1 1; -1 -1]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        L = size(crbc_interface_right,1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            crbc_dwdn_right,
                            crbc_interface_right,
                            q,
                            bulk_interface_right,
                            Complex{FT}.(SA[1 1; 1 -1]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        KernelAbstractions.synchronize(backend)

                        NVTX.@range "face rhs" begin
                        crbc_face_rhs!(
                            crbc_dq_top,
                            crbc_q_top,
                            crbc_data_top,
                            crbc_dwdn_top,
                            crbc_dqdtan_top,
                            crbc_grid_top,
                            crbc_invwJ_top,
                            crbc_DT_top,
                            crbc_materialparams_top,
                            a,
                            sig,
                            bc_top,
                            cm,
                            "top",
                        )

                        crbc_face_rhs!(
                            crbc_dq_bottom,
                            crbc_q_bottom,
                            crbc_data_bottom,
                            crbc_dwdn_bottom,
                            crbc_dqdtan_bottom,
                            crbc_grid_bottom,
                            crbc_invwJ_bottom,
                            crbc_DT_bottom,
                            crbc_materialparams_bottom,
                            a,
                            sig,
                            bc_bottom,
                            cm,
                            "bottom",
                        )

                        crbc_face_rhs!(
                            crbc_dq_left,
                            crbc_q_left,
                            crbc_data_left,
                            crbc_dwdn_left,
                            crbc_dqdtan_left,
                            crbc_grid_left,
                            crbc_invwJ_left,
                            crbc_DT_left,
                            crbc_materialparams_left,
                            a,
                            sig,
                            bc_left,
                            cm,
                            "left",
                        )

                        crbc_face_rhs!(
                            crbc_dq_right,
                            crbc_q_right,
                            crbc_data_right,
                            crbc_dwdn_right,
                            crbc_dqdtan_right,
                            crbc_grid_right,
                            crbc_invwJ_right,
                            crbc_DT_right,
                            crbc_materialparams_right,
                            a,
                            sig,
                            bc_right,
                            cm,
                            "right",
                        )
                        end

                        KernelAbstractions.synchronize(backend)

                        L = size(crbc_interface_left, 1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            data_lr,
                            bulk_interface_left,
                            crbc_dwdn_left,
                            crbc_interface_left,
                            Complex{FT}.(SA[-0.5 -0.5; 0.5 -0.5]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        L = size(crbc_interface_right, 1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            data_lr,
                            bulk_interface_right,
                            crbc_dwdn_right,
                            crbc_interface_right,
                            Complex{FT}.(SA[0.5 0.5; 0.5 -0.5]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        L = size(crbc_interface_top, 1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            data_tb,
                            bulk_interface_top,
                            crbc_dwdn_top,
                            crbc_interface_top,
                            Complex{FT}.(SA[0.5 0.5; 0.5*im -0.5*im]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )

                        L = size(crbc_interface_bottom, 1)
                        workgroup = 32
                        blocks = cld(L, workgroup)
                        idxcopyP!(backend, workgroup)(
                            data_tb,
                            bulk_interface_bottom,
                            crbc_dwdn_bottom,
                            crbc_interface_bottom,
                            Complex{FT}.(SA[-0.5 -0.5; 0.5*im -0.5*im]),
                            Val(workgroup),
                            Val(L);
                            ndrange = workgroup * blocks,
                        )
                    end

                    KernelAbstractions.synchronize(backend)

                    NVTX.@range "bulk rhs" begin
                    rhs!(dq, q, grid, data_lr, data_tb, invwJ, DT, materialparams, bc, cm)
                    end

                    KernelAbstractions.synchronize(backend)
                    @. q += RKB[stage] * dt * dq

                    if BC == :rbc
                        @. crbc_q_top += RKB[stage] * dt * crbc_dq_top
                        @. crbc_q_bottom += RKB[stage] * dt * crbc_dq_bottom
                        @. crbc_q_left += RKB[stage] * dt * crbc_dq_left
                        @. crbc_q_right += RKB[stage] * dt * crbc_dq_right
                        @. crbc_q_corner1 += RKB[stage] * dt * crbc_dq_corner1
                        @. crbc_q_corner2 += RKB[stage] * dt * crbc_dq_corner2
                        @. crbc_q_corner3 += RKB[stage] * dt * crbc_dq_corner3
                        @. crbc_q_corner4 += RKB[stage] * dt * crbc_dq_corner4
                    end
                end
                end #NVTX range stage
                time += dt

                vtk_output(step, time, q, materialparams, grid)
                energy_output(step, energy, q, grid)
            end # elapsed time

            cli_status_output(step, energy, numberofsteps, elapsed_per_step)
        end # time step for loop
    end #total elapsed time


    vtk_output(numberofsteps, timeend, q, materialparams, grid)
    energy_output(numberofsteps, energy, q, grid)
    cli_status_output(numberofsteps, energy, numberofsteps, elapsed)

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

        #jl # TODO add sum to GridArray so the following reduction is on the device
        err = sqrt(sum(Adapt.adapt(Array, wJ .* abs2.(q .- qexact))))

        #TODO output error here as with the emperical test
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

AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array
let

    FT = Float64
    vtkdir = "vtk_semidg_dirac_2d$(K)x$(K)_L$(Lout)_$(String(BC))_$(timeend)"

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if outputvtk || bigrun
        if CUDA.functional()
            CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
            CUDA.allowscalar(true) #FIXME: HERE
        end

        if rank == 0
            configdata = """Configuration:
                precision           = $FT
                array type          = $AT

                emperical conv test = $empericalconvergetest
                conv numlevels      = $(empericalconvergetest ? numlevels : "N/A")
                outputvtk           = $outputvtk
                outputdir           = $(outputvtk ? vtkdir : "N/A")
                base                = $K
                level refine        = $Lout
                polynomial order    = $N
                flux type           = $(String(flux))
                matparam            = $(String(matcoef))
                m                   = $mtemp
                bc type             = $(String(BC))
                x periodic          = $xperiodic
                y periodic          = $yperiodic
                crbc data file      = $crbcdatafile
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
            run(FT, AT, N, K, Lout,crbcdatafile; outputvtk = outputvtk, progress = outputprogress, vtkdir, comm)
        if outputvtk
            println("Energy % change:", energy[end] / energy[1])
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
        @assert mtemp > 0 "Exact solution requires m > 0"

        for l = 1:numlevels
            L = l - 1
            totalcells = (K * 2^L, K * 2^L)
            dofs = prod(totalcells) * prod(1 .+ N)
            _, _, _, err[l] =
                run(FT, AT, N, K, L,crbcdatafile; outputvtk = outputvtk, progress = outputprogress, vtkdir, comm)

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
            final_course, _, _, _ = run(FT, AT, N, K, L,crbcdatafile; outputvtk = false, comm)
            final_fine, grid, _, _ = run(FT, AT, N, K, L + 1,crbcdatafile; outputvtk = false, comm)

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
