module ShallowWater
export ShallowWaterLaw

import ..BalanceLaws
using ..BalanceLaws: avg, roe_avg, constants
using StaticArrays
using LinearAlgebra: I

struct ShallowWaterLaw{FT,D,S,C,P} <: BalanceLaws.AbstractBalanceLaw{FT,D,S,C}
    problem::P
    function ShallowWaterLaw{FT,D}(;
        grav = 10,
        problem::P = BalanceLaws.DummyProblem(),
    ) where {FT,D,P}
        S = 2 + D
        C = (grav = FT(grav),)
        new{FT,D,S,C,P}(problem)
    end
end

function varsindices(law::ShallowWaterLaw)
    S = BalanceLaws.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρθ = S
    return ix_ρ, ix_ρu⃗, ix_ρθ
end

function unpackstate(law::ShallowWaterLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρθ = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρθ]
end

function BalanceLaws.flux(law::ShallowWaterLaw, q, aux)
    ρ, ρu⃗, ρθ = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    p = constants(law).grav * ρ ^ 2 / 2

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + p * I
    fρθ = u⃗ * ρθ

    hcat(fρ, fρu⃗, fρθ)
end

function BalanceLaws.wavespeed(law::ShallowWaterLaw, n⃗, q, aux)
    ρ, ρu⃗, ρθ = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + sqrt(constants(law).grav * ρ)
end

function BalanceLaws.entropy(law::ShallowWaterLaw, q, aux)
    ρ, ρu⃗, ρθ = unpackstate(law, q)
    grav = constants(law).grav
    ρu⃗' * ρu⃗ / 2ρ + grav * ρ ^ 2 / 2
end

function BalanceLaws.entropyvariables(law::ShallowWaterLaw, q, aux)
    FT = eltype(law)
    ρ, ρu⃗, ρθ = unpackstate(law, q)
    grav = constants(law).grav
    vρ = -ρu⃗' * ρu⃗ / (2 * ρ ^ 2) + grav * ρ
    vρu⃗ = ρu⃗ / ρ
    vρθ = FT(0)
    SVector(vρ, vρu⃗..., vρθ)
end

function BalanceLaws.surfaceflux(
    ::BalanceLaws.RoeFlux,
    law::ShallowWaterLaw,
    n⃗,
    q⁻,
    aux⁻,
    q⁺,
    aux⁺,
)
    g = constants(law).grav
    f⁻ = BalanceLaws.flux(law, q⁻, aux⁻)
    f⁺ = BalanceLaws.flux(law, q⁺, aux⁺)

    ρ⁻, ρu⃗⁻, ρθ⁻ = unpackstate(law, q⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    p⁻ = g * ρ⁻ ^ 2 / 2
    c⁻ = sqrt(g * ρ⁻)

    ρ⁺, ρu⃗⁺, ρθ⁺ = unpackstate(law, q⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    p⁺ = g * ρ⁺ ^ 2 / 2
    c⁺ = sqrt(g * ρ⁺)

    ρ = sqrt(ρ⁻ * ρ⁺)
    u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
    θ = roe_avg(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)

    uₙ = u⃗' * n⃗

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu⃗' * n⃗

    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² / 2
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² / 2
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    fp_ρ = (w1 + w2 + w3) / 2
    fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) + w2 * (u⃗ + c * n⃗) + w3 * u⃗ + w4 * (Δu⃗ - Δuₙ * n⃗)) / 2
    fp_ρθ = ((w1 + w2) * θ + w5) / 2

    (f⁻ + f⁺)' * n⃗ / 2 - SVector(fp_ρ, fp_ρu⃗..., fp_ρθ)
end

function BalanceLaws.twopointflux(
    ::BalanceLaws.EntropyConservativeFlux,
    law::ShallowWaterLaw,
    q₁,
    _,
    q₂,
    _,
)
    FT = eltype(law)
    ρ₁, ρu⃗₁, ρθ₁ = unpackstate(law, q₁)
    ρ₂, ρu⃗₂, ρθ₂ = unpackstate(law, q₂)

    u⃗₁ = ρu⃗₁ / ρ₁
    θ₁ = ρθ₁ / ρ₁

    u⃗₂ = ρu⃗₂ / ρ₂
    θ₂ = ρθ₂ / ρ₂

    ρ_avg = avg(ρ₁, ρ₂)
    ρ²_avg = avg(ρ₁ ^ 2, ρ₂ ^ 2)
    u⃗_avg = avg(u⃗₁, u⃗₂)
    ρu⃗_avg = avg(ρu⃗₁, ρu⃗₂)
    θ_avg = avg(θ₁, θ₂)

    fρ = ρu⃗_avg
    fρu⃗ = ρu⃗_avg * u⃗_avg' + constants(law).grav * (ρ_avg ^ 2 - ρ²_avg / 2) * I
    fρθ = ρu⃗_avg * θ_avg

    hcat(fρ, fρu⃗, fρθ)
end
end
