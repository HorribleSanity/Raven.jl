using Raven
using Raven.BalanceLaws
using Raven.BalanceLaws.Advection

using Test
using Printf
using StaticArrays: SVector
using LinearAlgebra: norm

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV, "RAVEN_INTEGRATION_TESTING", "false")))
end

function wave(law, x⃗, t)
    ρ = 2 + sin(π * (sum(x⃗) - sum(constants(law).u⃗) * t))
    SVector(ρ)
end

function run(A, FT, N, K; volume_form = WeakForm())
    Nq = N + 1

    law = AdvectionLaw{FT,3}()

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)
    v1d = range(FT(-1), stop = FT(1), length = K + 1)
    coarsegrid = brick((v1d, v1d, v1d), (true, true, true))
    gm = GridManager(cell, coarsegrid)
    grid = generate(gm)

    dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RusanovFlux())

    cfl = FT(1 // 4)
    dt = cfl * step(v1d) / N / norm(constants(law).u⃗)
    timeend = FT(0.7)

    q = GridArray(undef, law, grid)
    q .= wave.(Ref(law), points(grid), FT(0))

    @info @sprintf """Starting
    N           = %d
    K           = %d
    volume_form = %s
    norm(q)     = %.16e
    """ N K volume_form weightednorm(dg, q)

    odesolver = LSRK54(dg, q, dt)
    solve!(q, timeend, odesolver)

    qexact = GridArray(undef, law, grid)
    qexact .= wave.(Ref(law), points(grid), timeend)
    errf = weightednorm(dg, q .- qexact)

    @info @sprintf """Finished
    norm(q)      = %.16e
    norm(q - qe) = %.16e
    """ weightednorm(dg, q) errf
    errf
end

let
    A = Array
    FT = Float64
    N = 4

    expected_error = Dict()

    #form, lev
    expected_error[WeakForm(), 1] = 5.2720204799800677e-04
    expected_error[WeakForm(), 2] = 1.6784569227037394e-05

    expected_error[FluxDifferencingForm(CentralFlux()), 1] = 5.2720204799799615e-04
    expected_error[FluxDifferencingForm(CentralFlux()), 2] = 1.6784569226995235e-05

    nlevels = integration_testing ? 2 : 1

    @testset for volume_form in (WeakForm(), FluxDifferencingForm(CentralFlux()))
        errors = zeros(FT, nlevels)
        for l = 1:nlevels
            K = 5 * 2^(l - 1)
            errors[l] = run(A, FT, N, K; volume_form)
            @test errors[l] ≈ expected_error[volume_form, l]
        end
        if nlevels > 1
            rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
            @info "Convergence rates\n" * join(
                ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(nlevels-1)],
                "\n",
            )
            @test rates[end] ≈ N + 1 atol = 0.11
        end
    end
end
