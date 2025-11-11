module RavenDiffEqBaseExt
import Raven: semidiscretize, GridVectorView, GridArray
using Raven: norm
import DiffEqBase
import SciMLBase: ODEProblem

function rhs(dq::GridVectorView, q::GridVectorView, dg, t)
    dg(parent(dq), parent(q), t, increment = false)
end

function semidiscretize(dg, tspan)
    q = GridArray(undef, dg.law, dg.grid)
    _q = GridVectorView(q)

    return ODEProblem(rhs, _q, tspan, dg)
end

DiffEqBase.ODE_DEFAULT_NORM(u::GridVectorView, t) = norm(u)

# TODO: MPI.Allreduce(any(!isfinite, u))
DiffEqBase.INFINITE_OR_GIANT(u::GridVectorView) = false
end
