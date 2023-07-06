@testset "Sparse Arrays" begin
    S = sparse([1], [2], [3])
    G = Raven.GeneralSparseMatrixCSC(S)

    AT = CUDA.functional() ? CuArray : Array
    H = Adapt.adapt(AT, G)

    for A in (G, H)
        @test size(S) == size(A)
        @test SparseArrays.getcolptr(S) == Array(SparseArrays.getcolptr(A))
        @test rowvals(S) == Array(rowvals(A))
        @test nonzeros(S) == Array(nonzeros(A))
        @test nnz(S) == nnz(A)

        @static if VERSION >= v"1.7"
            ioS = IOBuffer()
            ioA = IOBuffer()
            show(ioS, S)
            show(ioA, A)
            @test take!(ioS) == take!(ioA)

            @test_nowarn show(IOBuffer(), MIME"text/plain"(), A)
        end
    end
end
