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
    cell = LobattoCell{Tuple{N,N},FT,AT}()

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
        @test isapprox(pts[fm.vmapM[1]], pts[fm.vmapP[1]])
        @test isapprox(pts[fm.vmapM[2]], pts[fm.vmapP[2]])

        @test isapprox(
            pts[fm.avmapM[:, 1:numcells(grid)]],
            pts[fm.avmapP[:, 1:numcells(grid)]],
        )
        @test isapprox(
            pts[fm.avmapM[fm.amapM[:, 1:numcells(grid)]]],
            pts[fm.avmapM[fm.amapP[:, 1:numcells(grid)]]],
        )

        for n in eachindex(fm.mapB, fm.vmapM)
            # Test that faces in mapB are on the boundary
            @test all(
                isapprox.(
                    one(FT),
                    map(x -> maximum(abs.(x)), pts[fm.vmapM[n][fm.mapB[n]]]),
                ),
            )

            # Test that faces not in mapB are not on the boundary
            Nf = N^(ndims(cell) - 1)
            mapBn = reshape(fm.mapB[n], (Nf, :))
            vmapMn = reshape(fm.vmapM[n], (Nf, :))
            boundaryfaces = fld1.(mapBn, Nf)[1, :]
            nonboundaryfaces = setdiff(1:size(vmapMn, 2), boundaryfaces)
            for f in nonboundaryfaces
                @test !all(isapprox.(one(FT), map(x -> maximum(abs.(x)), pts[vmapMn])))
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
    cell = LobattoCell{Tuple{N,N,N}}()

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
        @test isapprox(pts[fm.vmapM[1]], pts[fm.vmapP[1]])
        @test isapprox(pts[fm.vmapM[2]], pts[fm.vmapP[2]])
        @test isapprox(pts[fm.vmapM[3]], pts[fm.vmapP[3]])

        @test isapprox(
            pts[fm.avmapM[:, 1:numcells(grid)]],
            pts[fm.avmapP[:, 1:numcells(grid)]],
        )
        @test isapprox(
            pts[fm.avmapM[fm.amapM[:, 1:numcells(grid)]]],
            pts[fm.avmapM[fm.amapP[:, 1:numcells(grid)]]],
        )

        @test all(
            isapprox.(one(FT), map(x -> maximum(abs.(x)), pts[fm.vmapM[2][fm.mapB[2]]])),
        )
        @test all(
            isapprox.(one(FT), map(x -> maximum(abs.(x)), pts[fm.vmapM[3][fm.mapB[3]]])),
        )

        for n in eachindex(fm.mapB, fm.vmapM)
            # Test that faces in mapB are on the boundary
            @test all(
                isapprox.(
                    one(FT),
                    map(x -> maximum(abs.(x)), pts[fm.vmapM[n][fm.mapB[n]]]),
                ),
            )

            # Test that faces not in mapB are not on the boundary
            Nf = N^(ndims(cell) - 1)
            mapBn = reshape(fm.mapB[n], (Nf, :))
            vmapMn = reshape(fm.vmapM[n], (Nf, :))
            boundaryfaces = fld1.(mapBn, Nf)[1, :]
            nonboundaryfaces = setdiff(1:size(vmapMn, 2), boundaryfaces)
            for f in nonboundaryfaces
                @test !all(isapprox.(one(FT), map(x -> maximum(abs.(x)), pts[vmapMn])))
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
