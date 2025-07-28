function gridarrays_testsuite(AT, FT)
    let
        N = (3, 2)
        K = (2, 3)
        L = 1
        gm = GridManager(LobattoCell{FT,AT}(N...), Raven.brick(FT, K); min_level = L)
        grid = generate(gm)

        x = points(grid)
        @test x isa GridArray
        @test Raven.get_backend(x) == KernelAbstractions.get_backend(x)

        @test size(x) == (N..., prod(K) * 4^L)
        y = AT(x)
        @test y isa AT

        @test sum(x) ≈ sum(y)

        F = Raven.fieldindex(x)
        xp = parent(x)
        nonfielddims = SVector((1:(F-1))..., ((F+1):ndims(xp))...)
        perm = insert(nonfielddims, 1, F)
        xp = permutedims(xp, perm)
        @test reinterpret(reshape, FT, y) == xp

        @test GridArray{SVector{0,FT}}(undef, grid) isa GridArray
    end
end
