using ReadVTK

function grids_testsuite(AT, FT)
    rng = StableRNG(37)

    let
        N = (3, 2)
        K = (2, 3)
        coordinates =
            ntuple(d -> range(-one(FT), stop = one(FT), length = K[d] + 1), length(K))

        gm = GridManager(
            LobattoCell{Tuple{N...},FT,AT}(),
            Raven.brick(coordinates);
            min_level = 2,
        )

        indicator = rand(rng, (Raven.AdaptNone, Raven.AdaptRefine), length(gm))

        adapt!(gm, indicator)
        warp(x::SVector{2}) = SVector(
            x[1] + cospi(3x[2] / 2) * cospi(x[1] / 2) * cospi(x[2] / 2) / 5,
            x[2] + sinpi(3x[1] / 2) * cospi(x[1] / 2) * cospi(x[2] / 2) / 5,
        )
        grid = generate(warp, gm)

        @test grid isa Raven.Grid
        @test issparse(grid.continuoustodiscontinuous)

        mktempdir() do tmpdir
            vtk_grid("$tmpdir/grid", grid) do vtk
                vtk["CellNumber"] = (1:length(grid)) .+ offset(grid)
                P = toequallyspaced(referencecell(grid))
                x = P * points(grid)
                vtk["x"] = Adapt.adapt(Array, x)
            end
            @test isfile("$tmpdir/grid.pvtu")
            @test isdir("$tmpdir/grid")
            @test_nowarn VTKFile("$tmpdir/grid/grid_1.vtu")
        end
    end

    let
        N = (3, 3)
        R = 1

        coarse_grid = Raven.cubeshellgrid(R)

        gm = GridManager(LobattoCell{Tuple{N...},FT,AT}(), coarse_grid, min_level = 2)

        indicator = rand((Raven.AdaptNone, Raven.AdaptRefine), length(gm))
        adapt!(gm, indicator)

        grid = generate(gm)

        @test grid isa Raven.Grid
        @test issparse(grid.continuoustodiscontinuous)

        mktempdir() do tmpdir
            vtk_grid("$tmpdir/grid", grid) do vtk
                vtk["CellNumber"] = (1:length(grid)) .+ offset(grid)
                P = toequallyspaced(referencecell(grid))
                x = P * reshape(points(grid), size(P, 2), :)
                vtk["x"] = Adapt.adapt(Array, x)
            end
            @test isfile("$tmpdir/grid.pvtu")
            @test isdir("$tmpdir/grid")
            @test_nowarn VTKFile("$tmpdir/grid/grid_1.vtu")
        end
    end

    let
        N = (4, 4)
        K = (2, 1)
        coordinates =
            ntuple(d -> range(-one(FT), stop = one(FT), length = K[d] + 1), length(K))
        cell = LobattoCell{Tuple{N...},FT,AT}()
        gm = GridManager(cell, brick(coordinates, (true, true)))
        grid = generate(gm)
        @test all(boundarycodes(grid) .== 0)
    end

    let
        N = (3, 2, 4)
        K = (2, 3, 1)
        coordinates =
            ntuple(d -> range(-one(FT), stop = one(FT), length = K[d] + 1), length(K))

        gm = GridManager(
            LobattoCell{Tuple{N...},FT,AT}(),
            Raven.brick(coordinates);
            min_level = 1,
        )

        indicator = rand(rng, (Raven.AdaptNone, Raven.AdaptRefine), length(gm))

        adapt!(gm, indicator)
        warp(x::SVector{3}) = x

        grid = generate(warp, gm)

        @test grid isa Raven.Grid
        @test issparse(grid.continuoustodiscontinuous)

        mktempdir() do tmpdir
            vtk_grid("$tmpdir/grid", grid) do vtk
                vtk["CellNumber"] = (1:length(grid)) .+ offset(grid)
                P = toequallyspaced(referencecell(grid))
                x = P * points(grid)
                vtk["x"] = Adapt.adapt(Array, x)
            end
            @test isfile("$tmpdir/grid.pvtu")
            @test isdir("$tmpdir/grid")
            @test_nowarn VTKFile("$tmpdir/grid/grid_1.vtu")
        end
    end


    let
        cell = LobattoCell{Tuple{4,4},FT,AT}()

        vertices = [
            SVector{2,FT}(0, 0), # 1
            SVector{2,FT}(2, 0), # 2
            SVector{2,FT}(0, 2), # 3
            SVector{2,FT}(2, 2), # 4
            SVector{2,FT}(4, 0), # 5
            SVector{2,FT}(4, 2), # 6
        ]
        cells = [(1, 2, 3, 4), (4, 2, 6, 5)]
        cg = coarsegrid(vertices, cells)

        gm = GridManager(cell, cg)
        grid = generate(gm)

        @test grid isa Raven.Grid
        @test issparse(grid.continuoustodiscontinuous)

        mktempdir() do tmpdir
            vtk_grid("$tmpdir/grid", grid) do vtk
                vtk["CellNumber"] = (1:length(grid)) .+ Raven.offset(grid)
                P = toequallyspaced(referencecell(grid))
                x = P * points(grid)
                vtk["x"] = Adapt.adapt(Array, x)
            end
            @test isfile("$tmpdir/grid.pvtu")
            @test isdir("$tmpdir/grid")
            @test_nowarn VTKFile("$tmpdir/grid/grid_1.vtu")
        end
    end

    let
        cell = LobattoCell{Tuple{3,3,3},FT,AT}()

        vertices = [
            SVector{3,FT}(0, 0, 0), #  1
            SVector{3,FT}(2, 0, 0), #  2
            SVector{3,FT}(0, 2, 0), #  3
            SVector{3,FT}(2, 2, 0), #  4
            SVector{3,FT}(0, 0, 2), #  5
            SVector{3,FT}(2, 0, 2), #  6
            SVector{3,FT}(0, 2, 2), #  7
            SVector{3,FT}(2, 2, 2), #  8
            SVector{3,FT}(4, 0, 0), #  9
            SVector{3,FT}(4, 2, 0), # 10
            SVector{3,FT}(4, 0, 2), # 11
            SVector{3,FT}(4, 2, 2), # 12
        ]
        cells = [(1, 2, 3, 4, 5, 6, 7, 8), (4, 2, 10, 9, 8, 6, 12, 11)]
        cg = coarsegrid(vertices, cells)

        gm = GridManager(cell, cg)
        grid = generate(gm)

        @test grid isa Raven.Grid
        @test issparse(grid.continuoustodiscontinuous)

        mktempdir() do tmpdir
            vtk_grid("$tmpdir/grid", grid) do vtk
                vtk["CellNumber"] = (1:length(grid)) .+ Raven.offset(grid)
                P = toequallyspaced(referencecell(grid))
                x = P * points(grid)
                vtk["x"] = Adapt.adapt(Array, x)
            end
            @test isfile("$tmpdir/grid.pvtu")
            @test isdir("$tmpdir/grid")
            @test_nowarn VTKFile("$tmpdir/grid/grid_1.vtu")
        end
    end

    @testset "2D metrics" begin

        f(x) = SA[
            9*x[1]-(1+x[1])*x[2]^2+(x[1]-1)^2*(1-x[2]^2+x[2]^3),
            10*x[2]+x[1]*x[1]^3*(1-x[2])+x[1]^2*x[2]*(1+x[2]),
        ]
        f11(x) = 7 + x[2]^2 - 2 * x[2]^3 + 2 * x[1] * (1 - x[2]^2 + x[2]^3)
        f12(x) = -2 * (1 + x[1]) * x[2] + (-1 + x[1])^2 * x[2] * (-2 + 3 * x[2])
        f21(x) = -4 * x[1]^3 * (-1 + x[2]) + 2 * x[1] * x[2] * (1 + x[2])
        f22(x) = 10 - x[1] * x[1]^3 + x[1]^2 * (1 + 2 * x[2])

        fJ(x, dx1, dx2, l) = SA[
            f11(x)*(dx1/2^(l+1)) f12(x)*(dx2/2^(l+1))
            f21(x)*(dx1/2^(l+1)) f22(x)*(dx2/2^(l+1))
        ]

        coordinates = (
            range(-one(FT), stop = one(FT), length = 6),
            range(-one(FT), stop = one(FT), length = 5),
        )

        L = 10
        M = 12
        level = 2

        gm = GridManager(
            LobattoCell{Tuple{L,M},FT,AT}(),
            Raven.brick(coordinates);
            min_level = level,
        )

        unwarpedgrid = generate(gm)
        grid = generate(f, gm)

        wr, ws = weights_1d(referencecell(gm))
        ux = points(unwarpedgrid)
        uJ = fJ.(ux, step.(coordinates)..., level)
        uwJ = wr .* ws .* det.(uJ)
        uinvJ = inv.(uJ)
        invJ, wJ = components(first(volumemetrics(grid)))

        @test all(adapt(Array, wJ .≈ uwJ))
        @test all(adapt(Array, invJ .≈ uinvJ))

        wJinvG, wJ = components(last(volumemetrics(grid)))
        @test all(adapt(Array, wJ .≈ uwJ))
        g(x) = x * x'
        @test all(adapt(Array, wJinvG) .≈ adapt(Array, uwJ .* g.(uinvJ)))

        uJ = adapt(Array, uJ)
        wr = adapt(Array, wr)
        ws = adapt(Array, ws)
        sm, _ = surfacemetrics(grid)
        sm = adapt(Array, sm)

        n, wsJ = components(sm)
        a = n .* wsJ
        @test all(
            a[1:M, 1:numcells(grid)] ./ vec(ws) .≈
            map(g -> SA[-g[2, 2], g[1, 2]], uJ[1, :, :]),
        )
        @test all(
            a[M.+(1:M), :] ./ vec(ws) .≈ map(g -> SA[g[2, 2], -g[1, 2]], uJ[end, :, :]),
        )
        @test all(
            a[2M.+(1:L), 1:numcells(grid)] ./ vec(wr) .≈
            map(g -> SA[g[2, 1], -g[1, 1]], uJ[:, 1, :]),
        )
        @test all(
            a[2M.+L.+(1:L), 1:numcells(grid)] ./ vec(wr) .≈
            map(g -> SA[-g[2, 1], g[1, 1]], uJ[:, end, :]),
        )

        @test all(norm.(n) .≈ 1)

    end

    @testset "2D spherical shell" begin

        L = 6
        level = 1
        R = FT(3)

        gm = GridManager(
            # Polynomial orders need to be the same to match up faces.
            # We currently do not support different polynomial orders for
            # joining faces.
            LobattoCell{Tuple{L,L},FT,AT}(),
            Raven.cubeshellgrid(R);
            min_level = level,
        )

        grid = generate(gm)
        _, wJ = components(first(volumemetrics(grid)))

        @test sum(adapt(Array, wJ)) ≈ pi * R^2 * 4

        pts = points(grid)
        fm = facemaps(grid)

        pts = Adapt.adapt(Array, pts)
        fm = Adapt.adapt(Array, fm)

        @test isapprox(pts[fm.vmapM], pts[fm.vmapP])
        @test isapprox(pts[fm.vmapM[fm.mapM]], pts[fm.vmapM[fm.mapP]])
    end

    @testset "2D constant preserving" begin

        f(x) = SA[
            9*x[1]-(1+x[1])*x[2]^2+(x[1]-1)^2*(1-x[2]^2+x[2]^3),
            10*x[2]+x[1]*x[1]^3*(1-x[2])+x[1]^2*x[2]*(1+x[2]),
        ]

        coordinates = (
            range(-one(FT), stop = one(FT), length = 3),
            range(-one(FT), stop = one(FT), length = 5),
        )

        L = 3
        M = 4
        level = 0

        gm = GridManager(
            LobattoCell{Tuple{L,M},FT,AT}(),
            Raven.brick(coordinates);
            min_level = level,
        )

        grid = generate(f, gm)
        invJ, wJ = components(first(volumemetrics(grid)))
        invJ11, invJ21, invJ12, invJ22 = components(invJ)

        wr, ws = weights_1d(referencecell(gm))
        J = wJ ./ (wr .* ws)

        Dr, Ds = derivatives(referencecell(grid))

        cp1 = (Dr * (J .* invJ11) + Ds * (J .* invJ21))
        cp2 = (Dr * (J .* invJ12) + Ds * (J .* invJ22))
        @test norm(adapt(Array, cp1), Inf) < 100 * eps(FT)
        @test norm(adapt(Array, cp2), Inf) < 100 * eps(FT)

    end

    @testset "3D metrics" begin

        f(x) = SA[
            3x[1]+x[2]/5+x[3]/10+x[1]*x[2]^2*x[3]^3/3,
            4x[2]+x[1]^3*x[2]^2*x[3]/4,
            2x[3]+x[1]^2*x[2]*x[3]^3/2,
        ]
        f11(x) = 3 * oneunit(eltype(x)) + x[2]^2 * x[3]^3 / 3
        f12(x) = oneunit(eltype(x)) / 5 + 2 * x[1] * x[2] * x[3]^3 / 3
        f13(x) = oneunit(eltype(x)) / 10 + 3 * x[1] * x[2]^2 * x[3]^2 / 3
        f21(x) = 3 * x[1]^2 * x[2]^2 * x[3] / 4
        f22(x) = 4 * oneunit(eltype(x)) + 2 * x[1]^3 * x[2] * x[3] / 4
        f23(x) = x[1]^3 * x[2]^2 / 4
        f31(x) = 2 * x[1] * x[2] * x[3]^3 / 2
        f32(x) = x[1]^2 * x[3]^3 / 2
        f33(x) = 2 * oneunit(eltype(x)) + 3 * x[1]^2 * x[2] * x[3]^2 / 2

        fJ(x, dx1, dx2, dx3, l) = SA[
            f11(x)*(dx1/2^(l+1)) f12(x)*(dx2/2^(l+1)) f13(x)*(dx3/2^(l+1))
            f21(x)*(dx1/2^(l+1)) f22(x)*(dx2/2^(l+1)) f23(x)*(dx3/2^(l+1))
            f31(x)*(dx1/2^(l+1)) f32(x)*(dx2/2^(l+1)) f33(x)*(dx3/2^(l+1))
        ]

        coordinates = (
            range(-one(FT), stop = one(FT), length = 3),
            range(-one(FT), stop = one(FT), length = 2),
            range(-one(FT), stop = one(FT), length = 4),
        )

        L = 6
        M = 8
        N = 7
        level = 0

        gm = GridManager(
            LobattoCell{Tuple{L,M,N},FT,AT}(),
            Raven.brick(coordinates);
            min_level = level,
        )

        unwarpedgrid = generate(gm)
        grid = generate(f, gm)

        wr, ws, wt = weights_1d(referencecell(gm))
        ux = points(unwarpedgrid)
        uJ = fJ.(ux, step.(coordinates)..., level)
        uwJ = wr .* ws .* wt .* det.(uJ)
        uinvJ = inv.(uJ)
        invJ, wJ = components(first(volumemetrics(grid)))

        @test all(adapt(Array, wJ .≈ uwJ))
        @test all(adapt(Array, invJ .≈ uinvJ))

        wJinvG, wJ = components(last(volumemetrics(grid)))

        @test all(adapt(Array, wJ .≈ uwJ))
        g(x) = x * x'
        @test all(adapt(Array, wJinvG) .≈ adapt(Array, uwJ .* g.(uinvJ)))

        uJ = adapt(Array, uJ)
        uinvJ = adapt(Array, uinvJ)
        wr = adapt(Array, wr)
        ws = adapt(Array, ws)
        wt = adapt(Array, wt)
        sm, _ = surfacemetrics(grid)
        sm = adapt(Array, sm)
        b = det.(uJ) .* uinvJ

        n, wsJ = components(sm)
        a = n .* wsJ
        @test all(
            reshape(a[1:M*N, 1:numcells(grid)], (M, N, numcells(grid))) ./
            (vec(ws) .* vec(wt)') .≈ map(g -> -g[1, :], b[1, :, :, :]),
        )
        @test all(
            reshape(a[M*N.+(1:M*N), 1:numcells(grid)], (M, N, numcells(grid))) ./
            (vec(ws) .* vec(wt)') .≈ map(g -> g[1, :], b[end, :, :, :]),
        )
        @test all(
            reshape(a[2*M*N.+(1:L*N), 1:numcells(grid)], (L, N, numcells(grid))) ./
            (vec(wr) .* vec(wt)') .≈ map(g -> -g[2, :], b[:, 1, :, :]),
        )
        @test all(
            reshape(a[2*M*N.+L*N.+(1:L*N), 1:numcells(grid)], (L, N, numcells(grid))) ./
            (vec(wr) .* vec(wt)') .≈ map(g -> g[2, :], b[:, end, :, :]),
        )
        @test all(
            reshape(a[2*L*N+2*M*N.+(1:L*M), 1:numcells(grid)], (L, M, numcells(grid))) ./
            (vec(wr) .* vec(ws)') .≈ map(g -> -g[3, :], b[:, :, 1, :]),
        )
        @test all(
            reshape(
                a[2*L*N+2*M*N.+L*M.+(1:L*M), 1:numcells(grid)],
                (L, M, numcells(grid)),
            ) ./ (vec(wr) .* vec(ws)') .≈ map(g -> g[3, :], b[:, :, end, :]),
        )

        @test all(norm.(n) .≈ 1)

    end

    @testset "3D constant preserving" begin

        f(x) = SA[
            3x[1]+x[2]/5+x[3]/10+x[1]*x[2]^2*x[3]^3/3,
            4x[2]+x[1]^3*x[2]^2*x[3]/4,
            2x[3]+x[1]^2*x[2]*x[3]^3/2,
        ]

        coordinates = (
            range(-one(FT), stop = one(FT), length = 3),
            range(-one(FT), stop = one(FT), length = 5),
            range(-one(FT), stop = one(FT), length = 4),
        )

        L = 3
        M = 4
        N = 2
        level = 0

        gm = GridManager(
            LobattoCell{Tuple{L,M,N},FT,AT}(),
            Raven.brick(coordinates);
            min_level = level,
        )

        grid = generate(f, gm)
        invJ, wJ = components(first(volumemetrics(grid)))
        invJc = components(invJ)

        wr, ws, wt = weights_1d(referencecell(gm))
        J = wJ ./ (wr .* ws .* wt)

        Dr, Ds, Dt = derivatives(referencecell(grid))

        cp1 = (Dr * (J .* invJc[1]) + Ds * (J .* invJc[2]) + Dt * (J .* invJc[3]))
        cp2 = (Dr * (J .* invJc[4]) + Ds * (J .* invJc[5]) + Dt * (J .* invJc[6]))
        cp3 = (Dr * (J .* invJc[7]) + Ds * (J .* invJc[8]) + Dt * (J .* invJc[9]))

        @test norm(adapt(Array, cp1), Inf) < 100 * eps(FT)
        @test norm(adapt(Array, cp2), Inf) < 100 * eps(FT)
        @test norm(adapt(Array, cp3), Inf) < 100 * eps(FT)

    end

end
