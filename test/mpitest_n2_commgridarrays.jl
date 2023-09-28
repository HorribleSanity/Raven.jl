using CUDA
using MPI
using Test
using Raven
using Raven.StaticArrays
using Raven.Adapt

MPI.Init()

function test(::Type{FT}, ::Type{AT}) where {FT,AT}
    @testset "Communicate GridArray ($AT, $FT)" begin
        N = (3, 2)
        K = (2, 1)
        min_level = 1
        cell = LobattoCell{Tuple{N...},Float64,AT}()
        gm = GridManager(cell, Raven.brick(K...); min_level)
        grid = generate(gm)

        T = SVector{2,FT}
        cm = Raven.commmanager(T, nodecommpattern(grid); comm = Raven.comm(grid), tag = 1)
        A = GridArray{T}(undef, grid)

        nan = convert(FT, NaN)

        viewwithghosts(A) .= Ref((nan, nan))
        A .= Ref((MPI.Comm_rank(Raven.comm(grid)) + 1, 0))

        Raven.share!(A, cm)

        @test all(A.data[:, :, 1, :] .== MPI.Comm_rank(Raven.comm(grid)) + 1)
        @test all(A.data[:, :, 2, :] .== 0)

        @test MPI.Comm_size(Raven.comm(grid)) == 2
        r = MPI.Comm_rank(Raven.comm(grid))

        if r == 0
            @test all(A.datawithghosts[2:3, :, :, 5:6] .=== nan)
            @test all(A.datawithghosts[1, :, 1, 5:6] .== 2)
            @test all(A.datawithghosts[1, :, 2, 5:6] .== 0)
        elseif r == 1
            @test all(A.datawithghosts[1:2, :, :, 5:6] .=== nan)
            @test all(A.datawithghosts[3, :, 1, 5:6] .== 1)
            @test all(A.datawithghosts[3, :, 2, 5:6] .== 0)
        end
    end
end

function main()
    test(Float64, Array)

    if CUDA.functional()
        test(Float32, CuArray)
    end
end

main()
