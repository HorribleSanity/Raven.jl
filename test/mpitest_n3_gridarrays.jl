using CUDA
using CUDA.CUDAKernels
using MPI
using Test
using Raven
using Raven.StaticArrays

MPI.Init()

struct Stiffness <: FieldArray{Tuple{2,2},Float64,2}
    xx::Float64
    yx::Float64
    xy::Float64
    yy::Float64
end

function test(N, K, ::Type{FT}, ::Type{AT}) where {FT,AT}
    @testset "GridArray ($N, $AT, $FT)" begin
        cell = LobattoCell{Tuple{N...},Float64,AT}()
        gm = GridManager(cell, Raven.brick(K...); min_level = 1)
        grid = generate(gm)

        val = (E = SVector{3,Complex{FT}}(1, 3, 5), B = SVector{3,Complex{FT}}(7, 9, 11))
        T = typeof(val)
        A = GridArray{T}(undef, grid)
        @test eltype(A) == T

        A .= Ref(val)
        @test CUDA.@allowscalar A[1] == val

        Adata = AT(parent(A))
        colons = ntuple(_ -> Colon(), Val(length(N)))
        for i = 1:2:11
            @test all(Adata[colons..., i, :] .== i)
        end
        for i = 2:2:12
            @test all(Adata[colons..., i, :] .== 0)
        end

        val2 =
            (E = SVector{3,Complex{FT}}(2, 6, 10), B = SVector{3,Complex{FT}}(14, 18, 22))
        B = Raven.viewwithghosts(A)
        B .= Ref(val2)
        @test CUDA.@allowscalar B[end] == val2

        Adatadata = parentwithghosts(A)
        colons = ntuple(_ -> Colon(), Val(length(N)))
        for i = 1:2:11
            @test all(Adatadata[colons..., i, :] .== 2i)
        end
        for i = 2:2:12
            @test all(Adatadata[colons..., i, :] .== 0)
        end

        L = length(flatten(val))

        @test arraytype(A) <: AT
        @test Raven.showingghosts(A) == false
        @test Raven.fieldindex(A) == length(N) + 1
        @test Raven.fieldslength(A) == L

        @test size(A) == (size(cell)..., numcells(grid))
        @test sizewithghosts(A) == (size(cell)..., numcells(grid, Val(true)))
        @test size(parent(A)) == (size(cell)..., L, numcells(grid))
        @test size(parentwithghosts(A)) == (size(cell)..., L, numcells(grid, Val(true)))

        C = components(A)
        @test length(C) == 2
        @test C isa NamedTuple{(:E, :B)}

        @test C[1] isa GridArray{typeof(val[1])}
        @test C[2] isa GridArray{typeof(val[2])}

        D = components(C[1])
        @test length(D) == 3
        @test D[1] isa GridArray{Complex{FT}}
        @test D[2] isa GridArray{Complex{FT}}
        @test D[3] isa GridArray{Complex{FT}}

        E = components(D[1])
        @test length(E) == 2
        @test E[1] isa GridArray{FT}
        @test E[2] isa GridArray{FT}

        F = components(E[1])
        @test length(F) == 1
        @test F[1] isa GridArray{FT}

        A = GridArray{Stiffness}(undef, grid)
        @test eltype(A) == Stiffness
        @test components(A) isa NamedTuple{(:xx, :yx, :xy, :yy)}
    end
end

function main()
    test((2, 3), (2, 1), Float64, Array)
    test((2, 3, 2), (1, 2, 1), Float64, Array)

    if CUDA.functional()
        test((2, 3), (2, 1), Float32, CuArray)
        test((2, 3, 2), (1, 2, 1), Float32, CuArray)
    end
end

main()
