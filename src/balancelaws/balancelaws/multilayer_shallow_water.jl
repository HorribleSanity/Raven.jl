module MultilayerShallowWater
export MultilayerShallowWaterLaw

import ..BalanceLaws
using ..BalanceLaws: constants, avg
using StaticArrays
using StaticArrays: SUnitRange
using LinearAlgebra: I, norm
using KernelAbstractions.Extras: @unroll

export numberoflayers

struct MultilayerShallowWaterLaw{FT,D,S,C,P,L} <: BalanceLaws.AbstractBalanceLaw{FT,D,S,C}
    problem::P
    function MultilayerShallowWaterLaw{FT,D,L}(;
        ρ,
        g = 981 // 100,
        problem::P = BalanceLaws.DummyProblem(),
    ) where {FT,D,P,L}
        if length(ρ) != L || !isbits(ρ)
            throw(ArgumentError("The argument ρ=$ρ is not of length $L and/or isbits."))
        end
        S = (1 + D) * L
        g = FT(g)
        ρ = FT.(ρ)
        C = (; ρ, g)
        new{FT,D,S,C,P,L}(problem)
    end
end

numberoflayers(::MultilayerShallowWaterLaw{FT,D,S,C,P,L}) where {FT,D,S,C,P,L} = L

@generated function unpackstate(
    ::MultilayerShallowWaterLaw{FT,D,S,C,P,L},
    q,
) where {FT,D,S,C,P,L}
    h_args = [:(q[$l]) for l = 1:L]
    hu_args =
        [:(q[StaticArrays.SUnitRange($(L + (l - 1) * D + 1), $(L + l * D))]) for l = 1:L]

    quote
        @inbounds begin
            SVector{$L,$FT}($(h_args...)), SVector{$L,SVector{$D,$FT}}($(hu_args...))
        end
    end
end

function BalanceLaws.auxiliary(law::MultilayerShallowWaterLaw, x⃗)
    SVector(zero(eltype(law)))
end

function unpackaux(::MultilayerShallowWaterLaw, aux)
    @inbounds aux[1]
end

function totalenergy(law::MultilayerShallowWaterLaw, h, hu, b)
    L = numberoflayers(law)

    u = hu ./ h

    g = constants(law).g
    ρ = constants(law).ρ

    e = zero(eltype(h))
    @inbounds @unroll for i = 1:L
        e += ρ[i] * ((h[i] * u[i]' * u[i] + g * h[i]^2) / 2 + g * h[i] * b)
        for j = 1:(i-1)
            e += g * ρ[j] * h[j] * h[i]
        end
    end

    return e
end

# Calculate approximation for maximum wave speed for local Lax-Friedrichs-type
# dissipation as the maximum velocity magnitude plus the maximum speed of
# sound. This function uses approximate eigenvalues as there is no simple
# way to calculate them analytically.
#
function max_abs_speed_naive(law::MultilayerShallowWaterLaw, h₁, hu₁, h₂, hu₂)
    hm₁ = sum(h₁)
    hm₂ = sum(h₂)

    # Get the averaged velocity
    um₁ = sum(hu₁) / hm₁
    um₂ = sum(hu₂) / hm₂

    g = constants(law).g

    # Calculate the wave celerity on the left and right
    c₁ = sqrt(g * hm₁)
    c₂ = sqrt(g * hm₂)

    return max(norm(um₁), norm(um₂)) + max(c₁, c₂)
end

# Less "cautious", i.e., less overestimating `λ_max` compared to
# `max_abs_speed_naive`
function max_abs_speed(law::MultilayerShallowWaterLaw, h₁, hu₁, h₂, hu₂)
    hm₁ = sum(h₁)
    hm₂ = sum(h₂)

    # Get the averaged velocity
    um₁ = sum(hu₁) / hm₁
    um₂ = sum(hu₂) / hm₂

    g = constants(law).g

    # Calculate the wave celerity on the left and right
    c₁ = sqrt(g * hm₁)
    c₂ = sqrt(g * hm₂)

    return max(norm(um₁) + c₁, norm(um₂) + c₂)
end

function BalanceLaws.entropy(law::MultilayerShallowWaterLaw, q, aux)
    h, hu = unpackstate(law, q)
    b = first(unpackaux(law, aux))
    totalenergy(law, h, hu, b)
end

function BalanceLaws.entropyvariables(law::MultilayerShallowWaterLaw, q, aux)
    L = numberoflayers(law)

    h, hu = unpackstate(law, q)
    b = first(unpackaux(law, aux))

    ρ = constants(law).ρ
    g = constants(law).g

    u = hu ./ h

    w₁ = similar(h)
    @inbounds @unroll for i = 1:L
        w₁[i] = ρ[i] * (g * b - u[i]' * u[i] / 2)
        @unroll for j = 1:L
            w₁[i] += ρ[min(j, i)] * h[j] * g
        end
    end
    w₁ = SVector(w₁)
    w₂ = u .* ρ

    vcat(w₁, w₂...)
end

function BalanceLaws.twopointflux(
    ::BalanceLaws.EntropyConservativeFlux,
    law::MultilayerShallowWaterLaw,
    q₁,
    aux₁,
    q₂,
    aux₂,
)
    L = numberoflayers(law)
    ρ = constants(law).ρ
    g = constants(law).g

    h₁, hu₁ = unpackstate(law, q₁)
    h₂, hu₂ = unpackstate(law, q₂)

    b₁ = first(unpackaux(law, aux₁))
    b₂ = first(unpackaux(law, aux₂))

    u₁ = hu₁ ./ h₁
    u₂ = hu₂ ./ h₂

    hu_avg = avg(hu₁, hu₂)
    u_avg = avg(u₁, u₂)

    fh = hu_avg
    fhu = hu_avg .* transpose.(u_avg)

    Δh = h₂ - h₁
    Δb = b₂ - b₁

    nc_fhu = @. g * h₁ * Δb
    nc_fhu = MArray(nc_fhu)

    @inbounds @unroll for i = 1:L
        @unroll for j = 1:L
            if j <= i - 1
                nc_fhu[i] += g * h₁[i] * Δh[j] * (ρ[j] / ρ[i])
            else
                nc_fhu[i] += g * h₁[i] * Δh[j]
            end
        end
    end

    fhu = fhu .+ (nc_fhu ./ 2) .* Scalar(I)

    hcat(fh..., fhu...)
end

end
