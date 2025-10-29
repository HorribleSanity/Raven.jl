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

        xx = GridArray{eltype(x)}(
            parentwithghosts(x),
            Raven.sizewithoutghosts(x),
            sizewithghosts(x),
            Raven.comm(x),
            Raven.showingghosts(x),
            Raven.fieldindex(x),
        )

        @test typeof(xx) === typeof(x)
        @test Raven.comm(xx) === Raven.comm(x)
        @test Raven.parentwithghosts(xx) === Raven.parentwithghosts(x)
        @test Raven.sizewithoutghosts(xx) === Raven.sizewithoutghosts(x)
        @test Raven.sizewithghosts(xx) === Raven.sizewithghosts(x)
    end
end
