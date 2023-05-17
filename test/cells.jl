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
    @test (1, 4, 4) == size.(Raven.materializefaces(cell), 2)
    @test Raven.connectivity(cell) == adapt(
        AT,
        (
            ([1 4 7; 2 5 8; 3 6 9],),
            ([1, 4, 7], [3, 6, 9], [1, 2, 3], [7, 8, 9]),
            (1, 3, 7, 9),
        ),
    )
    @test Raven.connectivityoffsets(cell, Val(1)) == (0, 9)
    @test Raven.connectivityoffsets(cell, Val(2)) == (0, 3, 6, 9, 12)
    @test Raven.connectivityoffsets(cell, Val(3)) == (0, 1, 2, 3, 4)
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
    @test (1, 6, 12, 8) == size.(Raven.materializefaces(cell), 2)
    @test Raven.connectivity(cell)[1] == (adapt(AT, reshape(collect(1:24), 3, 4, 2)),)
    @test Raven.connectivity(cell)[2:end] == adapt(
        AT,
        (
            (
                [1 13; 4 16; 7 19; 10 22],
                [3 15; 6 18; 9 21; 12 24],
                [1 13; 2 14; 3 15],
                [10 22; 11 23; 12 24],
                [1 4 7 10; 2 5 8 11; 3 6 9 12],
                [13 16 19 22; 14 17 20 23; 15 18 21 24],
            ),
            (
                [1, 13],
                [3, 15],
                [10, 22],
                [12, 24],
                [1, 4, 7, 10],
                [3, 6, 9, 12],
                [13, 16, 19, 22],
                [15, 18, 21, 24],
                [1, 2, 3],
                [10, 11, 12],
                [13, 14, 15],
                [22, 23, 24],
            ),
            (1, 3, 10, 12, 13, 15, 22, 24),
        ),
    )
    @test Raven.connectivityoffsets(cell, Val(1)) == (0, 24)
    @test Raven.connectivityoffsets(cell, Val(2)) == (0, 8, 16, 22, 28, 40, 52)
    @test Raven.connectivityoffsets(cell, Val(3)) ==
          (0, 2, 4, 6, 8, 12, 16, 20, 24, 27, 30, 33, 36)
    @test Raven.connectivityoffsets(cell, Val(4)) == (0, 1, 2, 3, 4, 5, 6, 7, 8)

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
    @test (1, 2) == size.(Raven.materializefaces(cell), 2)
    @test Raven.connectivity(cell) == adapt(AT, (([1, 2, 3, 4, 5],), (1, 5)))
    @test Raven.connectivityoffsets(cell, Val(1)) == (0, 5)
    @test Raven.connectivityoffsets(cell, Val(2)) == (0, 1, 2)
end
