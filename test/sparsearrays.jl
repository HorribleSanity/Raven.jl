@testset "Sparse Arrays" begin
    S = sparse([1], [2], [3])
    G = Raven.GeneralSparseMatrixCSC(S)
    H = Adapt.adapt(Array, G)

    for A in (G, H)
        @test size(S) == size(A)
        @test SparseArrays.getcolptr(S) == SparseArrays.getcolptr(A)
        @test rowvals(S) == rowvals(A)
        @test nonzeros(S) == nonzeros(A)
    end
end
