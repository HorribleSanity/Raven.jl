using Raven.StaticArrays: size_to_tuple

function cells_testsuite(AT, FT)
    cell = LobattoCell{Tuple{3,3},FT,AT}()
    @test floattype(typeof(cell)) == FT
    @test arraytype(typeof(cell)) <: AT
    @test ndims(typeof(cell)) == 2
    @test size(typeof(cell)) == (3, 3)
    @test length(typeof(cell)) == 9
    @test floattype(cell) == FT
    @test arraytype(cell) <: AT
    @test ndims(cell) == 2
    @test size(cell) == (3, 3)
    @test length(cell) == 9
    @test sum(mass(cell)) .≈ 4
    @test mass(cell) isa Diagonal
    @test sum(facemass(cell)) .≈ 8
    @test facemass(cell) isa Diagonal
    @test faceoffsets(cell) == (0, 3, 6, 9, 12)
    D = derivatives(cell)
    @test Array(D[1] * points(cell)) ≈ fill(SVector(one(FT), zero(FT)), 9)
    @test Array(D[2] * points(cell)) ≈ fill(SVector(zero(FT), one(FT)), 9)
    D1d = derivatives_1d(cell)
    @test length(D1d) == ndims(cell)
    @test all(
        Array.(D1d) .==
        setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(FT))) + 2))) do
            Matrix{
                FT,
            }.(spectralderivative.(first.(legendregausslobatto.(BigFloat, size(cell)))))
        end,
    )
    h1d = tohalves_1d(cell)
    @test length(h1d) == ndims(cell)
    @test all(length.(h1d) .== 2)
    setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(FT))) + 2))) do
        p, _ = legendregausslobatto(BigFloat, 3)
        I1 = Matrix{FT}(spectralinterpolation(p, (p .- 1) / 2))
        I2 = Matrix{FT}(spectralinterpolation(p, (p .+ 1) / 2))
        @test Array(h1d[1][1]) == I1
        @test Array(h1d[1][2]) == I2
        @test Array(h1d[2][1]) == I1
        @test Array(h1d[2][2]) == I2
    end
    @test adapt(Array, cell) isa LobattoCell{S,FT,Array} where {S}

    S = Tuple{3,4,2}
    cell = LobattoCell{S,FT,AT}()
    @test floattype(cell) == FT
    @test arraytype(cell) <: AT
    @test Base.ndims(cell) == 3
    @test size(cell) == size_to_tuple(S)
    @test length(cell) == prod(size_to_tuple(S))
    @test sum(mass(cell)) .≈ 8
    @test mass(cell) isa Diagonal
    @test sum(facemass(cell)) .≈ 24
    @test facemass(cell) isa Diagonal
    @test faceoffsets(cell) == (0, 8, 16, 22, 28, 40, 52)
    D = derivatives(cell)
    @test Array(D[1] * points(cell)) ≈
          fill(SVector(one(FT), zero(FT), zero(FT)), prod(size_to_tuple(S)))
    @test Array(D[2] * points(cell)) ≈
          fill(SVector(zero(FT), one(FT), zero(FT)), prod(size_to_tuple(S)))
    @test Array(D[3] * points(cell)) ≈
          fill(SVector(zero(FT), zero(FT), one(FT)), prod(size_to_tuple(S)))
    D1d = derivatives_1d(cell)
    @test length(D1d) == ndims(cell)
    @test all(
        Array.(D1d) .==
        setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(FT))) + 2))) do
            Matrix{
                FT,
            }.(spectralderivative.(first.(legendregausslobatto.(BigFloat, size(cell)))))
        end,
    )
    h1d = tohalves_1d(cell)
    @test length(h1d) == ndims(cell)
    @test all(length.(h1d) .== 2)
    setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(FT))) + 2))) do
        for (n, N) in enumerate(size(cell))
            p, _ = legendregausslobatto(BigFloat, N)
            @test Array(h1d[n][1]) == Matrix{FT}(spectralinterpolation(p, (p .- 1) ./ 2))
            @test Array(h1d[n][2]) == Matrix{FT}(spectralinterpolation(p, (p .+ 1) ./ 2))
        end
    end

    cell = LobattoCell{Tuple{5},FT,AT}()
    @test floattype(cell) == FT
    @test arraytype(cell) <: AT
    @test Base.ndims(cell) == 1
    @test size(cell) == (5,)
    @test length(cell) == 5
    @test sum(mass(cell)) .≈ 2
    @test mass(cell) isa Diagonal
    @test sum(facemass(cell)) .≈ 2
    @test facemass(cell) isa Diagonal
    @test faceoffsets(cell) == (0, 1, 2)
    D = derivatives(cell)
    @test Array(D[1] * points(cell)) ≈ fill(SVector(one(FT)), 5)
    D1d = derivatives_1d(cell)
    @test length(D1d) == ndims(cell)
    @test all(
        Array.(D1d) .==
        setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(FT))) + 2))) do
            Matrix{
                FT,
            }.(spectralderivative.(first.(legendregausslobatto.(BigFloat, size(cell)))))
        end,
    )
    h1d = tohalves_1d(cell)
    @test length(h1d) == ndims(cell)
    @test all(length.(h1d) .== 2)
    setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(FT))) + 2))) do
        p, _ = legendregausslobatto(BigFloat, size(cell)[1])
        @test Array(h1d[1][1]) == Matrix{FT}(spectralinterpolation(p, (p .- 1) ./ 2))
        @test Array(h1d[1][2]) == Matrix{FT}(spectralinterpolation(p, (p .+ 1) ./ 2))
    end
end
