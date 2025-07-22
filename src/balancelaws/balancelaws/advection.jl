module Advection
export AdvectionLaw

import ..BalanceLaws
using ..BalanceLaws: constants
using StaticArrays: SVector

struct AdvectionLaw{FT,D,C,P} <: BalanceLaws.AbstractBalanceLaw{FT,D,1,C}
    problem::P
    function AdvectionLaw{FT,D}(
        u⃗ = ntuple(d->FT(1), D);
        problem::P = BalanceLaws.DummyProblem(),
    ) where {FT,D,P}
        new{FT,D,(;u⃗),P}(problem)
    end
end

BalanceLaws.flux(law::AdvectionLaw, q, aux) = SVector(constants(law).u⃗) * q'
BalanceLaws.wavespeed(law::AdvectionLaw, n⃗, q, aux) = abs(n⃗' * SVector(constants(law).u⃗))
BalanceLaws.entropy(law::AdvectionLaw, q, aux) = q' * q / 2
BalanceLaws.entropyvariables(law::AdvectionLaw, q, aux) = q
end
