@testset "Arrays" begin
    A = [13, 4, 5, 1, 5]

    B = Raven.numbercontiguous(Int32, A)
    @test eltype(B) == Int32
    @test B == [4, 2, 3, 1, 3]

    B = Raven.numbercontiguous(Int32, A, by = x -> -x)
    @test eltype(B) == Int32
    @test B == [1, 3, 2, 4, 2]
end
