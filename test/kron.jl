function kron_testsuite(AT, FT)
    rng = StableRNG(37)
    a = adapt(AT, rand(rng, FT, 3, 2))
    b = adapt(AT, rand(rng, FT, 4, 5))
    c = adapt(AT, rand(rng, FT, 1, 7))

    for args in (
        (a,),
        (a, Raven.Eye{FT}(5)),
        (Raven.Eye{FT}(2), b),
        (a, b),
        (Raven.Eye{FT}(3), Raven.Eye{FT}(2), c),
        (Raven.Eye{FT}(2), b, Raven.Eye{FT}(7)),
        (a, Raven.Eye{FT}(4), Raven.Eye{FT}(7)),
        (a, b, c),
    )
        K = adapt(AT, collect(Raven.Kron(adapt(Array, args))))
        d = adapt(AT, rand(SVector{2,FT}, size(K, 2), 12))
        e = adapt(AT, rand(SVector{2,FT}, size(K, 2)))
        @test Array(Raven.Kron(args) * e) ≈ Array(K) * Array(e)
        @test Array(Raven.Kron(args) * d) ≈ Array(K) * Array(d)

        g = adapt(AT, rand(FT, 4, size(K, 2), 6))
        gv1 = @view g[1, :, :]
        gv2 = @view g[1, :, 1]
        @test Array(Raven.Kron(args) * gv1) ≈ Array(K) * Array(gv1)
        @test Array(Raven.Kron(args) * gv2) ≈ Array(K) * Array(gv2)

        if isbits(FT)
            f = rand(rng, FT, size(K, 2), 3, 2)
            f = adapt(
                AT,
                reinterpret(reshape, SVector{2,FT}, PermutedDimsArray(f, (3, 1, 2))),
            )
            @test Array(Raven.Kron(args) * f) ≈ Array(K) * Array(f)
        end

        @test adapt(Array, Raven.Kron(args)) == Raven.Kron(adapt.(Array, args))
    end
end

function kron2dgridarray_testsuite(AT, FT)
    rng = StableRNG(37)
    A = adapt(AT, rand(rng, FT, 5, 2))
    B = adapt(AT, rand(rng, FT, 2, 3))

    cell = LobattoCell{FT,AT}(2, 3)
    gm = GridManager(cell, Raven.brick(2, 1); min_level = 1)
    grid = generate(gm)

    EntryType = SVector{2,FT}
    v = GridArray{EntryType}(undef, grid)
    v .= adapt(AT, rand(rng, EntryType, size(v)))

    for args in ((Raven.Eye{FT}(3), A), (B, Raven.Eye{FT}(2)), (B, A))
        K = adapt(AT, collect(Raven.Kron(adapt(Array, args))))
        Kv1 = adapt(Array, Raven.Kron(args) * v)
        dimsIn = (prod(size(v)[1:(end-1)]), size(v)[end])
        dimsOut = (size.(reverse(args), 1)..., size(v)[end])
        Kv2 = reshape(Array(K) * reshape(adapt(Array, v), dimsIn), dimsOut)
        @test isapprox(Kv1, Kv2)
    end
end

function kron3dgridarray_testsuite(AT, FT)
    rng = StableRNG(37)
    A = adapt(AT, rand(rng, FT, 5, 2))
    B = adapt(AT, rand(rng, FT, 2, 3))
    C = adapt(AT, rand(rng, FT, 3, 5))

    cell = LobattoCell{FT,AT}(2, 3, 5)
    gm = GridManager(cell, Raven.brick(2, 1, 1); min_level = 1)
    grid = generate(gm)

    EntryType = SVector{2,FT}
    v = GridArray{EntryType}(undef, grid)
    v .= adapt(AT, rand(rng, EntryType, size(v)))

    for args in (
        (Raven.Eye{FT}(5), Raven.Eye{FT}(3), A),
        (Raven.Eye{FT}(5), B, Raven.Eye{FT}(2)),
        (C, Raven.Eye{FT}(3), Raven.Eye{FT}(2)),
        (Raven.Eye{FT}(5), B, A),
        (C, Raven.Eye{FT}(3), A),
        (C, B, Raven.Eye{FT}(2)),
        (C, B, A),
    )
        K = adapt(AT, collect(Raven.Kron(adapt(Array, args))))
        Kv1 = adapt(Array, Raven.Kron(args) * v)
        dimsIn = (prod(size(v)[1:(end-1)]), size(v)[end])
        dimsOut = (size.(reverse(args), 1)..., size(v)[end])
        Kv2 = reshape(Array(K) * reshape(adapt(Array, v), dimsIn), dimsOut)
        @test isapprox(Kv1, Kv2)
    end
end
