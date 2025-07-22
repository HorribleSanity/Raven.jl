module Euler
export EulerLaw

import ..BalanceLaws
using ..BalanceLaws: avg, logavg, roe_avg, constants
using StaticArrays
using LinearAlgebra: I, norm, cross

struct EulerLaw{FT,D,S,C,P} <: BalanceLaws.AbstractBalanceLaw{FT,D,S,C}
    problem::P
    function EulerLaw{FT,D}(;
        γ = 7 // 5,
        problem::P = BalanceLaws.DummyProblem(),
    ) where {FT,D,P}
        S = 2 + D
        C = (γ = FT(γ),)
        new{FT,D,S,C,P}(problem)
    end
end

function varsindices(law::EulerLaw)
    S = BalanceLaws.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρe = S
    return ix_ρ, ix_ρu⃗, ix_ρe
end

function unpackstate(law::EulerLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρe = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe]
end

function pressure(law::EulerLaw, ρ, ρu⃗, ρe)
    γ = constants(law).γ
    (γ - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ)
end
function energy(law::EulerLaw, ρ, ρu⃗, p)
    γ = constants(law).γ
    p / (γ - 1) + ρu⃗' * ρu⃗ / 2ρ
end
function soundspeed(law::EulerLaw, ρ, p)
    γ = constants(law).γ
    sqrt(γ * p / ρ)
end
function soundspeed(law::EulerLaw, ρ, ρu⃗, ρe)
    soundspeed(law, ρ, pressure(law, ρ, ρu⃗, ρe))
end

function BalanceLaws.flux(law::EulerLaw, q, _)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    p = pressure(law, ρ, ρu⃗, ρe)

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + p * I
    fρe = u⃗ * (ρe + p)

    hcat(fρ, fρu⃗, fρe)
end

function BalanceLaws.wavespeed(law::EulerLaw, n⃗, q, aux)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + soundspeed(law, ρ, ρu⃗, ρe)
end

function BalanceLaws.entropy(law::EulerLaw, q, aux)
    ρ, ρu⃗, ρe = unpackstate(law, q)
    γ = constants(law).γ
    p = pressure(law, ρ, ρu⃗, ρe)
    s = log(p / ρ ^ γ)

    -ρ * s / (γ - 1)
end

function BalanceLaws.entropyvariables(law::EulerLaw, q, aux)
    ρ, ρu⃗, ρe = unpackstate(law, q)
    γ = constants(law).γ
    p = pressure(law, ρ, ρu⃗, ρe)
    s = log(p / ρ ^ γ)
    b = ρ / 2p
    u⃗ = ρu⃗ / ρ
    vρ = (γ - s) / (γ - 1) - u⃗' * u⃗ * b
    vρu⃗ = 2b * u⃗
    vρe = -2b

    SVector(vρ, vρu⃗..., vρe)
end

function BalanceLaws.surfaceflux(
    ::BalanceLaws.RoeFlux,
    law::EulerLaw,
    n⃗,
    q⁻,
    aux⁻,
    q⁺,
    aux⁺,
)
    f⁻ = BalanceLaws.flux(law, q⁻, aux⁻)
    f⁺ = BalanceLaws.flux(law, q⁺, aux⁺)

    ρ⁻, ρu⃗⁻, ρe⁻ = unpackstate(law, q⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻)
    h⁻ = e⁻ + p⁻ / ρ⁻
    c⁻ = soundspeed(law, ρ⁻, p⁻)

    ρ⁺, ρu⃗⁺, ρe⁺ = unpackstate(law, q⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺)
    h⁺ = e⁺ + p⁺ / ρ⁺
    c⁺ = soundspeed(law, ρ⁺, p⁺)

    ρ = sqrt(ρ⁻ * ρ⁺)
    u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
    h = roe_avg(ρ⁻, ρ⁺, h⁻, h⁺)
    c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)

    uₙ = u⃗' * n⃗

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δuₙ = Δu⃗' * n⃗

    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² / 2
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² / 2
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ

    fp_ρ = (w1 + w2 + w3) / 2
    fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) + w2 * (u⃗ + c * n⃗) + w3 * u⃗ + w4 * (Δu⃗ - Δuₙ * n⃗)) / 2
    fp_ρe =
        (
            w1 * (h - c * uₙ) +
            w2 * (h + c * uₙ) +
            w3 * u⃗' * u⃗ / 2 +
            w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)
        ) / 2

    (f⁻ + f⁺)' * n⃗ / 2 - SVector(fp_ρ, fp_ρu⃗..., fp_ρe)
end

function BalanceLaws.twopointflux(
    ::BalanceLaws.EntropyConservativeFlux,
    law::EulerLaw,
    q₁,
    _,
    q₂,
    _,
)
    FT = eltype(law)
    γ = constants(law).γ
    ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q₁)
    ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q₂)

    u⃗₁ = ρu⃗₁ / ρ₁
    p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁)
    b₁ = ρ₁ / 2p₁

    u⃗₂ = ρu⃗₂ / ρ₂
    p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂)
    b₂ = ρ₂ / 2p₂

    ρ_avg = avg(ρ₁, ρ₂)
    u⃗_avg = avg(u⃗₁, u⃗₂)
    b_avg = avg(b₁, b₂)

    u²_avg = avg(u⃗₁' * u⃗₁, u⃗₂' * u⃗₂)
    ρ_log = logavg(ρ₁, ρ₂)
    b_log = logavg(b₁, b₂)

    fρ = u⃗_avg * ρ_log
    fρu⃗ = u⃗_avg * fρ' + ρ_avg / 2b_avg * I
    fρe = (1 / (2 * (γ - 1) * b_log) - u²_avg / 2) * fρ + fρu⃗ * u⃗_avg

    hcat(fρ, fρu⃗, fρe)
end

function BalanceLaws.twopointflux(
    ::BalanceLaws.KennedyGruberFlux,
    law::EulerLaw,
    q₁,
    aux₁,
    q₂,
    aux₂,
)
    FT = eltype(law)
    ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q₁)
    ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q₂)

    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁)

    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂)

    ρ_avg = avg(ρ₁, ρ₂)
    u⃗_avg = avg(u⃗₁, u⃗₂)
    e_avg = avg(e₁, e₂)
    p_avg = avg(p₁, p₂)

    fρ = u⃗_avg * ρ_avg
    fρu⃗ = u⃗_avg * fρ' + p_avg * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)

    hcat(fρ, fρu⃗, fρe)
end

function BalanceLaws.surfaceflux(
    ::BalanceLaws.MatrixFlux,
    law::EulerLaw,
    n⃗,
    q⁻,
    aux⁻,
    q⁺,
    aux⁺,
)
    FT = eltype(law)
    γ = constants(law).γ
    ecflux = BalanceLaws.surfaceflux(
        BalanceLaws.EntropyConservativeFlux(),
        law,
        n⃗,
        q⁻,
        aux⁻,
        q⁺,
        aux⁺,
    )

    ρ⁻, ρu⃗⁻, ρe⁻ = unpackstate(law, q⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻)
    b⁻ = ρ⁻ / 2p⁻

    ρ⁺, ρu⃗⁺, ρe⁺ = unpackstate(law, q⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺)
    b⁺ = ρ⁺ / 2p⁺

    ρ_log = logavg(ρ⁻, ρ⁺)
    b_log = logavg(b⁻, b⁺)
    u⃗_avg = avg(u⃗⁻, u⃗⁺)
    p_avg = avg(ρ⁻, ρ⁺) / 2avg(b⁻, b⁺)
    u²_bar = 2 * norm(u⃗_avg) - avg(norm(u⃗⁻), norm(u⃗⁺))
    h_bar = γ / (2 * b_log * (γ - 1)) + u²_bar / 2
    c_bar = sqrt(γ * p_avg / ρ_log)

    u⃗mc = u⃗_avg - c_bar * n⃗
    u⃗pc = u⃗_avg + c_bar * n⃗
    u_avgᵀn = u⃗_avg' * n⃗

    v⁻ = BalanceLaws.entropyvariables(law, q⁻, aux⁻)
    v⁺ = BalanceLaws.entropyvariables(law, q⁺, aux⁺)
    Δv = v⁺ - v⁻

    λ1 = abs(u_avgᵀn - c_bar) * ρ_log / 2γ
    λ2 = abs(u_avgᵀn) * ρ_log * (γ - 1) / γ
    λ3 = abs(u_avgᵀn + c_bar) * ρ_log / 2γ
    λ4 = abs(u_avgᵀn) * p_avg

    Δv_ρ, Δv_ρu⃗, Δv_ρe = unpackstate(law, Δv)
    u⃗ₜ = u⃗_avg - u_avgᵀn * n⃗

    w1 = λ1 * (Δv_ρ + u⃗mc' * Δv_ρu⃗ + (h_bar - c_bar * u_avgᵀn) * Δv_ρe)
    w2 = λ2 * (Δv_ρ + u⃗_avg' * Δv_ρu⃗ + u²_bar / 2 * Δv_ρe)
    w3 = λ3 * (Δv_ρ + u⃗pc' * Δv_ρu⃗ + (h_bar + c_bar * u_avgᵀn) * Δv_ρe)

    Dρ = w1 + w2 + w3

    Dρu⃗ = (
        w1 * u⃗mc +
        w2 * u⃗_avg +
        w3 * u⃗pc +
        λ4 * (Δv_ρu⃗ - n⃗' * (Δv_ρu⃗) * n⃗ + Δv_ρe * u⃗ₜ)
    )

    Dρe = (
        w1 * (h_bar - c_bar * u_avgᵀn) +
        w2 * u²_bar / 2 +
        w3 * (h_bar + c_bar * u_avgᵀn) +
        λ4 * (u⃗ₜ' * Δv_ρu⃗ + Δv_ρe * (u⃗_avg' * u⃗_avg - u_avgᵀn ^ 2))
    )

    ecflux - SVector(Dρ, Dρu⃗..., Dρe) / 2
end
end
