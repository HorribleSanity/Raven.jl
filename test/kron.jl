function kron_testsuite(AT, FT)
    rng = StableRNG(37)
    a = adapt(AT, rand(rng, FT, 3, 2))
    b = adapt(AT, rand(rng, FT, 4, 5))
    c = adapt(AT, rand(rng, FT, 1, 7))

    for args in (
        (a,),
        (a, Eye{FT}(5)),
        (Eye{FT}(2), b),
        (a, b),
        (Eye{FT}(3), Eye{FT}(2), c),
        (Eye{FT}(2), b, Eye{FT}(7)),
        (a, Eye{FT}(4), Eye{FT}(7)),
        (a, b, c),
    )
        K = adapt(AT, collect(Harpy.Kron(adapt(Array, args))))
        d = adapt(AT, rand(SVector{2,FT}, size(K, 2), 12))
        e = adapt(AT, rand(SVector{2,FT}, size(K, 2)))
        @test Array(Harpy.Kron(args) * e) ≈ Array(K * e)
        @test Array(Harpy.Kron(args) * d) ≈ Array(K * d)

        g = adapt(AT, rand(FT, 4, size(K, 2), 6))
        gv1 = @view g[1, :, :]
        gv2 = @view g[1, :, 1]
        @test Array(Harpy.Kron(args) * gv1) ≈ Array(K * gv1)
        @test Array(Harpy.Kron(args) * gv2) ≈ Array(K * gv2)

        if isbits(FT)
            f = rand(rng, FT, size(K, 2), 3, 2)
            f = adapt(
                AT,
                reinterpret(reshape, SVector{2,FT}, PermutedDimsArray(f, (3, 1, 2))),
            )
            @test Array(Harpy.Kron(args) * f) ≈ Array(K * f)
        end

        @test adapt(Array, Harpy.Kron(args)) == Harpy.Kron(adapt.(Array, args))
    end
end
