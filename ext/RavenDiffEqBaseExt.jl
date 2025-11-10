module RavenDiffEqBaseExt
import Raven: semidiscretize, GridVectorView, GridArray
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
end
