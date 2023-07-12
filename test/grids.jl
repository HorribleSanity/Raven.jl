using ReadVTK

function grids_testsuite(AT, FT)
    rng = StableRNG(37)

    let
        N = (3, 2)
        K = (2, 3)
        coordinates =
            ntuple(d -> range(-one(FT), stop = one(FT), length = K[d] + 1), length(K))

        gm = GridManager(
            LobattoCell{Tuple{N...},Float64,AT}(),
            Raven.brick(K...; coordinates);
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
        nedge = 3

        coarse_grid = Raven.cubeshellgrid(R, nedge)

        gm = GridManager(
            LobattoCell{Tuple{N...},Float64,Array}(),
            coarse_grid,
            min_level = 2,
        )

        indicator = rand((Raven.AdaptNone, Raven.AdaptRefine), length(gm))
        adapt!(gm, indicator)

        function cubespherewarp(point::SVector{3})
            p = sortperm(abs.(point))
            point = point[p]

            ξ = π * point[2] / 4point[3]
            η = π * point[1] / 4point[3]

            y_x = tan(ξ)
            z_x = tan(η)

            x = point[3] / hypot(1, y_x, z_x)
            y = x * y_x
            z = x * z_x

            point = SVector(z, y, x)[sortperm(p)]

            return point
        end

        grid = generate(cubespherewarp, gm)

        @test grid isa Raven.Grid
        @test issparse(grid.continuoustodiscontinuous)

        mktempdir() do tmpdir
            vtk_grid("$tmpdir/grid", grid) do vtk
                vtk["CellNumber"] = (1:length(grid)) .+ offset(grid)
                P = toequallyspaced(referencecell(grid))
                x = P * reshape(points(grid), size(P, 2), :)
                vtk["x"] = collect(x)
            end
            @test isfile("$tmpdir/grid.pvtu")
            @test isdir("$tmpdir/grid")
            @test_nowarn VTKFile("$tmpdir/grid/grid_1.vtu")
        end
    end

    let
        N = (3, 2, 4)
        K = (2, 3, 1)
        coordinates =
            ntuple(d -> range(-one(FT), stop = one(FT), length = K[d] + 1), length(K))

        gm = GridManager(
            LobattoCell{Tuple{N...},Float64,AT}(),
            Raven.brick(K...; coordinates);
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
end
