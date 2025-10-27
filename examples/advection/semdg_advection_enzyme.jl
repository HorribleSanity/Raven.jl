using Enzyme

include("semdg_advection_2d.jl")

FT = Float64
AT = Array
N = (4, 4)
K = 4
L = 0

if !MPI.Initialized()
    MPI.Init()
end

comm = MPI.COMM_WORLD
solution = gaussian


cell = LobattoCell{FT,AT}((N .+ 1)...)
coordinates = ntuple(_ -> range(FT(0), stop = FT(2π), length = K + 1), 2)
periodicity = (true, true)
gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
grid = generate(meshwarp, gm)


# initialize state
q = solution.(points(grid), FT(0))

# storage for RHS
dq = similar(q)
dq .= 0

# precompute inverse of weights × Jacobian
_, _, wJ = components(first(volumemetrics(grid)))
invwJ = inv.(wJ)
# precompute derivative transpose
DT = transpose.(derivatives_1d(cell))

cm = commmanager(eltype(q), nodecommpattern(grid); comm)

cell = referencecell(grid)

dRdX, _, wJ = components(first(volumemetrics(grid)))
n, _, wsJ = components(first(surfacemetrics(grid)))
fm = facemaps(grid)

rhs!(dq, q, (cell, dRdX, wJ, n, wsJ, fm), invwJ, DT, cm)

# # compute gradient of rhs! at q
# δq = Enzyme.make_zero(q)
# δdq = Enzyme.make_zero(dq)
# Enzyme.autodiff(
#     Enzyme.set_runtime_activity(Forward),
#     # Forward,
#     rhs!,
#     Duplicated(dq, δdq),
#     Duplicated(q, δq),
#     Const((cell, dRdX, wJ, n, wsJ, fm)),
#     Const(invwJ),
#     Const(DT),
#     Const(cm),
# )

# δdq =  J(q) * δq 
# δq  = δdq * transpose(J)

# Enzyme.autodiff(
#     Reverse,
#     rhs!,
#     Duplicated(dq, δdq),
#     Duplicated(q, δq),
#     Const(grid),
#     Const(invwJ),
#     Const(DT),
#     Const(cm),
# )

using Ariadne

# function F(dq, q, (cell, dRdX, wJ, n, wsJ, fm, invwJ, DT, cm)) 
#     rhs!(dq, q, (cell, dRdX, wJ, n, wsJ, fm), invwJ, DT, cm)
#     return dq
# end

# J = Ariadne.JacobianOperator(F, dq, q, (cell, dRdX, wJ, n, wsJ, fm, invwJ, DT, cm))

# size(J)
# uu = similar(q)
# u = similar(q)

using LinearAlgebra

function LinearAlgebra.mul!(out::GridArray, J::Ariadne.JacobianOperator, v::GridArray)
    # @show J.p[3]
    autodiff(
        Forward,
        Ariadne.maybe_duplicated(J.f, J.f′), Const,
        Duplicated(J.res, out),
        Duplicated(J.u, v),
        Ariadne.maybe_duplicated(J.p, J.p′)
    )
    return nothing
end
# mul!(u, J, uu)


# collect(J)

# using EnzymeTestUtils

p = (cell, dRdX, wJ, n, wsJ, fm, invwJ, DT, cm)
# function EnzymeTestUtils.rand_tangent(rng, k::typeof(p))
#     return Enzyme.make_zero(k)
# end



# test_forward(F, Duplicated, (dq, Duplicated), (q, Duplicated), (p, Duplicated),
#     runtime_activity=false)


include(joinpath(dirname(pathof(Ariadne)), "..", "examples", "implicit.jl"))

function G_Euler2!(res, uₙ, Δt, f!, du, u, p, t)
    f!(du, u, p, t)

    for i in eachindex(res)
        res[i] = uₙ[i] + Δt * du[i] - u[i]
    end
    # res .= uₙ .+ Δt .* du .- u
    return nothing
end

function rhs2(dq, q,  (cell, dRdX, wJ, n, wsJ, fm, invwJ, DT, cm), t)
    dq .= 0
    rhs!(dq, q, (cell, dRdX, wJ, n, wsJ, fm), invwJ, DT, cm)
    return nothing
end
# J = jacobian(G_Euler2!, rhs2, q, p, 0.1, 0.0)

timeend = 0.02
Δt = 0.01
ts = 0.0:Δt:timeend

using Krylov
Krylov.ktypeof(x::GridArray{T}) where T = typeof(x)
Krylov.kcopy!(n::Integer, y::GridArray{T}, x::GridArray{T}) where T = copyto!(y, x)
Krylov.knorm(n::Integer, x::GridArray{T}) where T = norm(x)
Krylov.kdivcopy!(n::Integer, y::GridArray{T}, x::GridArray{T}, s::T) where T = (y .= x ./ s)
Krylov.kdot(n::Integer, x::GridArray{T}, y::GridArray{T}) where T = dot(x, y)
Krylov.kaxpy!(n::Integer, s::T, y::GridArray{T}, x::GridArray{T}) where T = axpy!(s, x, y) #(y .+= a .* x)

q = solution.(points(grid), FT(0))
solve(G_Euler2!, rhs2, q, p, Δt, ts;
   algo=:gmres, verbose = 1, krylov_kwargs = (; verbose = 1, reorthogonalization = true),
)


J = jacobian(G_Euler2!, rhs2, q, p, Δt, 0.0)

_u = deepcopy(J.u)
_p = deepcopy(J.p)


out = similar(J.res)
vv = zero(J.u)
mul!(out, J, vv)

p[1]
_p[1]

# J.u

real_J = Ariadne.collect(J)

@assert _u == J.u
@assert _p == J.p

norm(J.u)

J.f(J.res, J.u, J.p)
norm(J.res)

x = real_J \ -vec(J.res)
_x, stats = gmres(real_J, -vec(J.res), verbose=1)
x ≈ _x

norm(J.res)
norm(J.u)

kc = KrylovConstructor(J.res)
workspace = krylov_workspace(:gmres, kc)

norm(J.res)
norm(J.u)

J.f(J.res, J.u, J.p)
norm(J.res)

_u = deepcopy(J.u)
_p = deepcopy(J.p)

J.f(J.res, _u, _p)
norm(J.res)

krylov_solve!(workspace, J, -copy(J.res), verbose = 1)
workspace.stats

workspace.x 


J.f(J.res, J.u, J.p)
norm(J.res)


J.u == _u
J.p[1] == _p[1]
J.p[2] == _p[2]
J.p[3] == _p[3]


J.f(J.res, _u, _p)
norm(J.res)




qexact = solution.(points(grid), timeend)
errf = sqrt(MPI.Allreduce(sum(Adapt.adapt(Array, wJ .* (q .- qexact) .^ 2)), +, comm))
