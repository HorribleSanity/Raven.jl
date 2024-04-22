using CUDA
using CUDA.CUDAKernels
using MPI
using Test
using Raven
using Raven.StaticArrays
using LinearAlgebra
using Raven.Adapt

MPI.Init()

struct Stiffness <: FieldArray{Tuple{2,2},Float64,2}
    xx::Float64
    yx::Float64
    xy::Float64
    yy::Float64
end

function test(N, K, ::Type{FT}, ::Type{AT}) where {FT,AT}
    @testset "GridArray ($N, $AT, $FT)" begin
        minlvl = 1
        cell = LobattoCell{Float64,AT}(N...)
        gm = GridManager(cell, Raven.brick(K...); min_level = minlvl)
        grid = generate(gm)

        A = GridArray(undef, grid)
        @test A isa GridArray{Float64}
        @test arraytype(A) <: AT

        T = NamedTuple{(:E, :B),Tuple{SVector{3,Complex{FT}},SVector{3,Complex{FT}}}}
        A = GridArray{T}(undef, grid)
        @test eltype(A) == T

        val = (E = SA[1, 3, 5], B = SA[7, 9, 11])
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

        val = (E = SA[2, 1, 3], B = SA[0, 2, 1])
        fill!(A, val)
        @test CUDA.@allowscalar A[1] == val

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

        cval = convert(T, val)
        L = length(flatten(cval))

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

        @test C[1] isa GridArray{typeof(cval[1])}
        @test C[2] isa GridArray{typeof(cval[2])}

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

        val3 = (
            E = SVector{3,Complex{FT}}(1 + 1im, 1 + 1im, 1 + 1im),
            B = SVector{3,Complex{FT}}(1 + 1im, 1 + 1im, 1 + 1im),
        )
        L = length(flatten(cval))
        B = Raven.viewwithghosts(A)
        B .= Ref(val3)
        normA = sqrt(FT(L * prod(N) * prod(K) * (2^(length(K) * minlvl))))
        @test isapprox(norm(A), normA)

        A = GridArray{Stiffness}(undef, grid)
        @test eltype(A) == Stiffness
        @test components(A) isa NamedTuple{(:xx, :yx, :xy, :yy)}

        A = GridArray{FT}(undef, grid)
        A .= FT(2)
        B = 1 ./ A
        @test all(adapt(Array, (B .== FT(0.5))))

        B = copy(A)
        @test all(adapt(Array, (B .== A)))

        C = AT{FT}(undef, size(A))
        C .= -zero(FT)
        @test all(adapt(Array, (A .== (A .+ C))))
        @test all(adapt(Array, (A .== (C .+ A))))
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
