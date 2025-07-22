export CentralFlux
export RusanovFlux
export EntropyConservativeFlux
export KennedyGruberFlux
export RoeFlux
export MatrixFlux

avg(s⁻, s⁺) = (s⁻ + s⁺) / 2
function logavg(a, b)
    ζ = a / b
    f = (ζ - 1) / (ζ + 1)
    u = f^2
    ϵ = eps(eltype(u))

    if u < ϵ
        F = @evalpoly(u, one(u), one(u) / 3, one(u) / 5, one(u) / 7, one(u) / 9)
    else
        F = log(ζ) / 2f
    end

    (a + b) / 2F
end
roe_avg(ρ⁻, ρ⁺, s⁻, s⁺) = (sqrt(ρ⁻) * s⁻ + sqrt(ρ⁺) * s⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

abstract type AbstractNumericalFlux end
function twopointflux end
function surfaceflux(
    flux::AbstractNumericalFlux,
    law::AbstractBalanceLaw,
    n⃗,
    q⁻,
    aux⁻,
    q⁺,
    aux⁺,
)
    twopointflux(flux, law, q⁻, aux⁻, q⁺, aux⁺)' * n⃗
end

struct CentralFlux <: AbstractNumericalFlux end
function twopointflux(::CentralFlux, law::AbstractBalanceLaw, q₁, aux₁, q₂, aux₂)
    (flux(law, q₁, aux₁) + flux(law, q₂, aux₂)) / 2
end

struct RusanovFlux <: AbstractNumericalFlux end
function surfaceflux(::RusanovFlux, law::AbstractBalanceLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    fc = surfaceflux(CentralFlux(), law, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    ws⁻ = wavespeed(law, n⃗, q⁻, aux⁻)
    ws⁺ = wavespeed(law, n⃗, q⁺, aux⁺)
    fc - max(ws⁻, ws⁺) * (q⁺ - q⁻) / 2
end

struct EntropyConservativeFlux <: AbstractNumericalFlux end
struct KennedyGruberFlux <: AbstractNumericalFlux end
struct RoeFlux <: AbstractNumericalFlux end
struct MatrixFlux <: AbstractNumericalFlux end
