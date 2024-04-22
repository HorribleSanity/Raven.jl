using CUDA
using CUDA.CUDAKernels
using MPI
using Test
using Raven
using Raven.StaticArrays
using Raven.P4estTypes
using Raven.SparseArrays

MPI.Init()

let
    FT = Float64
    AT = Array

    N = 4
    cell = LobattoCell{FT,AT}(N, N)

    # 4--------5--------6
    # |        |        |
    # |        |        |
    # |        |        |
    # |        |        |
    # 1--------2--------3

    vertices = [
        SVector{2,FT}(-1, -1), # 1
        SVector{2,FT}(0, -1),  # 2
        SVector{2,FT}(1, -1),  # 3
        SVector{2,FT}(-1, 1),  # 4
        SVector{2,FT}(0, 1),   # 5
        SVector{2,FT}(1, 1),   # 6
    ]

    for cells in [[(1, 2, 4, 5), (2, 3, 5, 6)], [(1, 2, 4, 5), (5, 6, 2, 3)]]
        min_level = 0
        cg = coarsegrid(vertices, cells)
        gm = GridManager(cell, cg; min_level)
        grid = generate(gm)

        A = continuoustodiscontinuous(grid)
        pts = points(grid, Val(true))
        rows = rowvals(A)
        vals = nonzeros(A)
        _, n = size(A)

        ci = CartesianIndices(pts)

        for j = 1:n
            x = pts[rows[first(nzrange(A, j))]]
            for ii in nzrange(A, j)
                p = pts[rows[ii]]
                @test x ≈ p || (all(isnan.(x)) && all(isnan.(p)))

                # make sure all points that have all NaNs are in the
                # ghost layer
                if (all(isnan.(x)) && all(isnan.(p)))
                    i = rows[ii]
                    @test fld1(i, length(cell)) > numcells(grid)
                end
            end
        end

        cp = nodecommpattern(grid)
        commcells = communicatingcells(grid)
        noncommcells = noncommunicatingcells(grid)
        for i in cp.sendindices
            q = fld1(i, length(cell))
            @test q ∈ commcells
        end
        @test noncommcells == setdiff(0x1:numcells(grid), commcells)

        fm = facemaps(grid)
        @test isapprox(
            pts[fm.vmapM[:, 1:numcells(grid)]],
            pts[fm.vmapP[:, 1:numcells(grid)]],
        )
        @test isapprox(
            pts[fm.vmapM[fm.mapM[:, 1:numcells(grid)]]],
            pts[fm.vmapM[fm.mapP[:, 1:numcells(grid)]]],
        )

        bc = boundarycodes(grid)
        vmapM = reshape(fm.vmapM, (N, 4, :))
        for q in numcells(grid)
            for f = 1:4
                if bc[f, q] == 1
                    @test all(
                        isapprox.(one(FT), map(x -> maximum(abs.(x)), pts[vmapM[:, f, q]])),
                    )
                else
                    @test !all(
                        isapprox.(one(FT), map(x -> maximum(abs.(x)), pts[vmapM[:, f, q]])),
                    )
                end
            end
        end

        indicator =
            map(tid -> (tid == 1 ? Raven.AdaptNone : Raven.AdaptRefine), trees(grid))
        adapt!(gm, indicator)
        grid = generate(gm)
        fm = facemaps(grid)
        tohalves = tohalves_1d(cell)
        pts = points(grid, Val(true))
        fgdims = ((2,), (1,))
        _, ncsm = surfacemetrics(grid)
        for fg in eachindex(fm.vmapNC)
            for i = 1:last(size(fm.vmapNC[fg]))
                ppts = pts[fm.vmapNC[fg][:, 1, i]]
                c1pts = pts[fm.vmapNC[fg][:, 2, i]]
                c2pts = pts[fm.vmapNC[fg][:, 3, i]]

                @assert isapprox(tohalves[fgdims[fg][1]][1] * ppts, c1pts)
                @assert isapprox(tohalves[fgdims[fg][1]][2] * ppts, c2pts)
            end

            if length(ncsm[fg]) > 0
                n, wsJ = components(ncsm[fg])
                @test all(n[:, 1, :] .≈ Ref(SA[1, 0]))
                @test all(n[:, 2:end, :] .≈ Ref(SA[-1, 0]))
                @test sum(wsJ[:, 1, :]) .≈ sum(wsJ[:, 2:end, :])
            end
        end
    end
end

let
    FT = Float64
    AT = Array

    N = 3
    cell = LobattoCell(N, N, N)

    #    10-------11-------12
    #    /|       /|       /|
    #   / |      / |      / |
    #  /  |     /  |     /  |
    # 7--------8--------9   |
    # |   4----|---5----|---6
    # |  /     |  /     |  /
    # | /      | /      | /
    # |/       |/       |/
    # 1--------2--------3
    #
    vertices = [
        SVector{3,FT}(-1, -1, -1), # 1
        SVector{3,FT}(0, -1, -1),  # 2
        SVector{3,FT}(1, -1, -1),  # 3
        SVector{3,FT}(-1, 1, -1),  # 4
        SVector{3,FT}(0, 1, -1),   # 5
        SVector{3,FT}(1, 1, -1),   # 6
        SVector{3,FT}(-1, -1, 1),  # 7
        SVector{3,FT}(0, -1, 1),   # 8
        SVector{3,FT}(1, -1, 1),   # 9
        SVector{3,FT}(-1, 1, 1),   #10
        SVector{3,FT}(0, 1, 1),    #12
        SVector{3,FT}(1, 1, 1),    #13
    ]

    for cells in [
        #     2--------6      55        4--------2
        #    /|       /|               /|       /|
        #   / |      / |              / |      / |
        #  /  |     /  |             /  |     /  |
        # 4--------8   |  59        8--------6   |
        # |   1----|---5      54    |   3----|---1
        # |  /     |  /             |  /     |  /
        # | /      | /              | /      | /
        # |/       |/               |/       |/
        # 3--------7      58        7--------5
        [(4, 10, 1, 7, 5, 11, 2, 8), (6, 12, 5, 11, 3, 9, 2, 8)],
        #
        #     2--------6      55        7--------5
        #    /|       /|               /|       /|
        #   / |      / |              / |      / |
        #  /  |     /  |             /  |     /  |
        # 4--------8   |  59        3--------1   |
        # |   1----|---5      54    |   8----|---6
        # |  /     |  /             |  /     |  /
        # | /      | /              | /      | /
        # |/       |/               |/       |/
        # 3--------7      58        4--------2
        [(4, 10, 1, 7, 5, 11, 2, 8), (9, 3, 8, 2, 12, 6, 11, 5)],
        #
        #     2--------6      55        8--------6
        #    /|       /|               /|       /|
        #   / |      / |              / |      / |
        #  /  |     /  |             /  |     /  |
        # 4--------8   |  59        7--------5   |
        # |   1----|---5      54    |   4----|---2
        # |  /     |  /             |  /     |  /
        # | /      | /              | /      | /
        # |/       |/               |/       |/
        # 3--------7      58        3--------1
        [(4, 10, 1, 7, 5, 11, 2, 8), (3, 6, 2, 5, 9, 12, 8, 11)],
        #
        #     2--------6      55        3--------1
        #    /|       /|               /|       /|
        #   / |      / |              / |      / |
        #  /  |     /  |             /  |     /  |
        # 4--------8   |  59        4--------2   |
        # |   1----|---5      54    |   7----|---5
        # |  /     |  /             |  /     |  /
        # | /      | /              | /      | /
        # |/       |/               |/       |/
        # 3--------7      58        8--------6
        [(4, 10, 1, 7, 5, 11, 2, 8), (12, 9, 11, 8, 6, 3, 5, 2)],
        #
        #     2--------6      55       5--------7
        #    /|       /|              /|       /|
        #   / |      / |             / |      / |
        #  /  |     /  |            /  |     /  |
        # 4--------8   |  59       6--------8   |
        # |   1----|---5      54   |   1----|---3
        # |  /     |  /            |  /     |  /
        # | /      | /             | /      | /
        # |/       |/              |/       |/
        # 3--------7      58       2--------4
        [(4, 10, 1, 7, 5, 11, 2, 8), (5, 2, 6, 3, 11, 8, 12, 9)],
        #
        #     2--------6      55       5--------1
        #    /|       /|              /|       /|
        #   / |      / |             / |      / |
        #  /  |     /  |            /  |     /  |
        # 4--------8   |  59       8--------3   |
        # |   1----|---5      54   |   6----|---2
        # |  /     |  /            |  /     |  /
        # | /      | /             | /      | /
        # |/       |/              |/       |/
        # 3--------7      58       7--------4
        [(4, 10, 1, 7, 5, 11, 2, 8), (12, 6, 9, 3, 11, 5, 8, 2)],
        #
        #     2--------6      55       8--------4
        #    /|       /|              /|       /|
        #   / |      / |             / |      / |
        #  /  |     /  |            /  |     /  |
        # 4--------8   |  59       6--------2   |
        # |   1----|---5      54   |   7----|---3
        # |  /     |  /            |  /     |  /
        # | /      | /             | /      | /
        # |/       |/              |/       |/
        # 3--------7      58       5--------1
        [(4, 10, 1, 7, 5, 11, 2, 8), (3, 9, 6, 12, 2, 8, 5, 11)],
        #
        #     2--------6      55       6--------2
        #    /|       /|              /|       /|
        #   / |      / |             / |      / |
        #  /  |     /  |            /  |     /  |
        # 4--------8   |  59       5--------1   |
        # |   1----|---5      54   |   8----|---4
        # |  /     |  /            |  /     |  /
        # | /      | /             | /      | /
        # |/       |/              |/       |/
        # 3--------7      58       7--------3
        [(4, 10, 1, 7, 5, 11, 2, 8), (9, 12, 3, 6, 8, 11, 2, 5)],
    ]

        min_level = 1
        cg = coarsegrid(vertices, cells)
        gm = GridManager(cell, cg; min_level)

        grid = generate(gm)

        A = continuoustodiscontinuous(grid)
        pts = points(grid, Val(true))

        rows = rowvals(A)
        vals = nonzeros(A)
        _, n = size(A)

        ci = CartesianIndices(pts)

        for j = 1:n
            x = pts[rows[first(nzrange(A, j))]]
            for ii in nzrange(A, j)
                p = pts[rows[ii]]
                @test x ≈ p || (all(isnan.(x)) && all(isnan.(p)))

                # make sure all points that have all NaNs are in the
                # ghost layer
                if (all(isnan.(x)) && all(isnan.(p)))
                    i = rows[ii]
                    @test fld1(i, length(cell)) > numcells(grid)
                end
            end
        end

        cp = nodecommpattern(grid)
        commcells = communicatingcells(grid)
        noncommcells = noncommunicatingcells(grid)
        for i in cp.sendindices
            q = fld1(i, length(cell))
            @test q ∈ commcells
        end
        @test noncommcells == setdiff(0x1:numcells(grid), commcells)

        fm = facemaps(grid)
        @test isapprox(
            pts[fm.vmapM[:, 1:numcells(grid)]],
            pts[fm.vmapP[:, 1:numcells(grid)]],
        )
        @test isapprox(
            pts[fm.vmapM[fm.mapM[:, 1:numcells(grid)]]],
            pts[fm.vmapM[fm.mapP[:, 1:numcells(grid)]]],
        )

        bc = boundarycodes(grid)
        vmapM = reshape(fm.vmapM, (N, N, 6, :))
        for q in numcells(grid)
            for f = 1:6
                if bc[f, q] == 1
                    @test all(
                        isapprox.(
                            one(FT),
                            map(x -> maximum(abs.(x)), pts[vmapM[:, :, f, q]]),
                        ),
                    )
                else
                    @test !all(
                        isapprox.(
                            one(FT),
                            map(x -> maximum(abs.(x)), pts[vmapM[:, :, f, q]]),
                        ),
                    )
                end
            end
        end

        indicator =
            map(tid -> (tid == 1 ? Raven.AdaptNone : Raven.AdaptRefine), trees(grid))
        adapt!(gm, indicator)
        grid = generate(gm)
        fm = facemaps(grid)
        tohalves = tohalves_1d(cell)
        pts = points(grid, Val(true))
        fgdims = ((2, 3), (1, 3), (1, 2))
        _, ncsm = surfacemetrics(grid)
        for fg in eachindex(fm.vmapNC)
            for i = 1:last(size(fm.vmapNC[fg]))
                ppts = pts[fm.vmapNC[fg][:, :, 1, i]]
                c1pts = pts[fm.vmapNC[fg][:, :, 2, i]]
                c2pts = pts[fm.vmapNC[fg][:, :, 3, i]]
                c3pts = pts[fm.vmapNC[fg][:, :, 4, i]]
                c4pts = pts[fm.vmapNC[fg][:, :, 5, i]]

                @assert isapprox(
                    tohalves[fgdims[fg][1]][1] * ppts * tohalves[fgdims[fg][2]][1]',
                    c1pts,
                )
                @assert isapprox(
                    tohalves[fgdims[fg][1]][2] * ppts * tohalves[fgdims[fg][2]][1]',
                    c2pts,
                )
                @assert isapprox(
                    tohalves[fgdims[fg][1]][1] * ppts * tohalves[fgdims[fg][2]][2]',
                    c3pts,
                )
                @assert isapprox(
                    tohalves[fgdims[fg][1]][2] * ppts * tohalves[fgdims[fg][2]][2]',
                    c4pts,
                )
            end
            if length(ncsm[fg]) > 0
                n, wsJ = components(ncsm[fg])
                @test all(n[:, :, 1, :] .≈ Ref(SA[1, 0, 0]))
                @test all(n[:, :, 2:end, :] .≈ Ref(SA[-1, 0, 0]))
                @test sum(wsJ[:, :, 1, :]) .≈ sum(wsJ[:, :, 2:end, :])
            end
        end
    end
end
