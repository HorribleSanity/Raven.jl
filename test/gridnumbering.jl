@testset "Grid Numbering" begin
    function isisomorphic(a, b)
        f = Dict{eltype(a),eltype(b)}()
        g = Dict{eltype(b),eltype(a)}()

        for i in eachindex(a, b)
            if a[i] in keys(f)
                if f[a[i]] != b[i]
                    @error "$(a[i]) -> $(f[a[i]]) exists when adding $(a[i]) -> $(b[i])"
                    return false
                end
            else
                f[a[i]] = b[i]
            end

            if b[i] in keys(g)
                if g[b[i]] != a[i]
                    @error "$(g[b[i]]) <- $(b[i]) exists when adding $(a[i]) <- $(b[i])"
                    return false
                end
            else
                g[b[i]] = a[i]
            end
        end

        return true
    end

    function istranspose(ctod, dtoc)
        rows = rowvals(ctod)
        vals = nonzeros(ctod)
        _, n = size(ctod)
        for j = 1:n
            for k in nzrange(ctod, j)
                if vals[k] == false
                    return false
                end
                i = rows[k]
                if j != dtoc[i]
                    return false
                end
            end
        end

        return true
    end

    let
        # Coarse Grid
        #   y
        #   ^
        # 4 |  7------8------9
        #   |  |      |      |
        #   |  |      |      |
        #   |  |      |      |
        # 2 |  3------4------6
        #   |  |      |      |
        #   |  |      |      |
        #   |  |      |      |
        # 0 |  1------2------5
        #   |
        #   +------------------> x
        #      0      2      4
        vertices = [
            SVector(0.0, 0.0), # 1
            SVector(2.0, 0.0), # 2
            SVector(0.0, 2.0), # 3
            SVector(2.0, 2.0), # 4
            SVector(4.0, 0.0), # 5
            SVector(4.0, 2.0), # 6
            SVector(0.0, 4.0), # 7
            SVector(2.0, 4.0), # 8
            SVector(4.0, 4.0), # 9
        ]
        cells = [
            (1, 2, 3, 4), # 1
            (6, 4, 5, 2), # 2
            (3, 4, 7, 8), # 3
            (4, 6, 8, 9), # 4
        ]
        cg = coarsegrid(vertices, cells)

        forest = P4estTypes.pxest(Raven.connectivity(cg))
        P4estTypes.refine!(forest; refine = (_, tid, _) -> tid == 4)
        P4estTypes.balance!(forest)
        ghost = P4estTypes.ghostlayer(forest)
        nodes = P4estTypes.lnodes(forest; ghost, degree = 3)
        P4estTypes.expand!(ghost, forest, nodes)

        quadranttoglobalids = Raven.materializequadranttoglobalid(forest, ghost)
        @test quadranttoglobalids == 1:7

        quadrantcommpattern = Raven.materializequadrantcommpattern(forest, ghost)
        @test quadrantcommpattern.recvindices == 8:7
        @test quadrantcommpattern.recvranks == []
        @test quadrantcommpattern.recvrankindices == []
        @test quadrantcommpattern.sendindices == 8:7
        @test quadrantcommpattern.sendranks == []
        @test quadrantcommpattern.sendrankindices == []


        #               21  32  33    33  36  37
        #               18  30  31    31  34  35
        #                9  24  25    25  28  29
        #
        # 19  20  21    21  24  25    25  28  29
        # 16  17  18    18  22  23    23  26  27
        #  7   8   9     9  11  10     9  11  10
        #
        #  7   8   9     9  11  10
        #  4   5   6     6  13  12
        #  1   2   3     3  15  14
        (dtoc_degree_3_local, dtoc_degree_3_global) =
            Raven.materializedtoc(forest, ghost, nodes, quadrantcommpattern, MPI.COMM_WORLD)
        @test dtoc_degree_3_local == dtoc_degree_3_global
        @test dtoc_degree_3_local == P4estTypes.unsafe_element_nodes(nodes) .+ 0x1

        cell_degree_3 = LobattoCell{Tuple{4,4},Float64,Array}()
        dtoc_degree_3 =
            Raven.materializedtoc(cell_degree_3, dtoc_degree_3_local, dtoc_degree_3_global)
        @test isisomorphic(dtoc_degree_3, dtoc_degree_3_global)

        ctod_degree_3 = Raven.materializectod(dtoc_degree_3)
        @test ctod_degree_3 isa AbstractSparseMatrix
        @test istranspose(ctod_degree_3, dtoc_degree_3)

        quadranttolevel = Int8[
            P4estTypes.level.(Iterators.flatten(forest))
            P4estTypes.level.(P4estTypes.ghosts(ghost))
        ]
        parentnodes = Raven.materializeparentnodes(
            cell_degree_3,
            ctod_degree_3,
            quadranttoglobalids,
            quadranttolevel,
        )
        Np = length(cell_degree_3)
        for lid in eachindex(parentnodes)
            pid = parentnodes[lid]
            qlid = cld(lid, Np)
            qpid = cld(pid, Np)

            @test quadranttolevel[qpid] <= quadranttolevel[qlid]
            if quadranttolevel[qlid] == quadranttolevel[qpid]
                @test quadranttoglobalids[qlid] >= quadranttoglobalids[qpid]
            end
            @test dtoc_degree_3[pid] == dtoc_degree_3[lid]
        end

        GC.@preserve nodes begin
            @test Raven.materializequadranttofacecode(nodes) ==
                  P4estTypes.unsafe_face_code(nodes)
        end
    end

    let
        # Coarse Grid
        # z = 0:
        #   y
        #   ^
        # 4 |  7------8------9
        #   |  |      |      |
        #   |  |      |      |
        #   |  |      |      |
        # 2 |  3------4------6
        #   |  |      |      |
        #   |  |      |      |
        #   |  |      |      |
        # 0 |  1------2------5
        #   |
        #   +------------------> x
        #      0      2      4
        #
        # z = 2:
        #   y
        #   ^
        # 4 | 16-----17-----18
        #   |  |      |      |
        #   |  |      |      |
        #   |  |      |      |
        # 2 | 12-----13-----15
        #   |  |      |      |
        #   |  |      |      |
        #   |  |      |      |
        # 0 | 10-----11-----14
        #   |
        #   +------------------> x
        #      0      2      4
        vertices = [
            SVector(0.0, 0.0, 0.0), #  1
            SVector(2.0, 0.0, 0.0), #  2
            SVector(0.0, 2.0, 0.0), #  3
            SVector(2.0, 2.0, 0.0), #  4
            SVector(4.0, 0.0, 0.0), #  5
            SVector(4.0, 2.0, 0.0), #  6
            SVector(0.0, 4.0, 0.0), #  7
            SVector(2.0, 4.0, 0.0), #  8
            SVector(4.0, 4.0, 0.0), #  9
            SVector(0.0, 0.0, 2.0), # 10
            SVector(2.0, 0.0, 2.0), # 11
            SVector(0.0, 2.0, 2.0), # 12
            SVector(2.0, 2.0, 2.0), # 13
            SVector(4.0, 0.0, 2.0), # 14
            SVector(4.0, 2.0, 2.0), # 15
            SVector(0.0, 4.0, 2.0), # 16
            SVector(2.0, 4.0, 2.0), # 17
            SVector(4.0, 4.0, 2.0), # 18
        ]
        cells = [
            (1, 2, 3, 4, 10, 11, 12, 13), # 1
            (6, 4, 5, 2, 15, 13, 14, 11), # 2
            (3, 4, 7, 8, 12, 13, 16, 17), # 3
            (4, 6, 8, 9, 13, 15, 17, 18), # 4
        ]
        cg = coarsegrid(vertices, cells)

        forest = P4estTypes.pxest(Raven.connectivity(cg))
        P4estTypes.refine!(forest; refine = (_, tid, _) -> tid == 4)
        P4estTypes.balance!(forest)
        ghost = P4estTypes.ghostlayer(forest)
        nodes = P4estTypes.lnodes(forest; ghost, degree = 3)
        P4estTypes.expand!(ghost, forest, nodes)

        quadranttoglobalids = Raven.materializequadranttoglobalid(forest, ghost)
        @test quadranttoglobalids == 1:11

        quadrantcommpattern = Raven.materializequadrantcommpattern(forest, ghost)
        @test quadrantcommpattern.recvindices == 12:11
        @test quadrantcommpattern.recvranks == []
        @test quadrantcommpattern.recvrankindices == []
        @test quadrantcommpattern.sendindices == 12:11
        @test quadrantcommpattern.sendranks == []
        @test quadrantcommpattern.sendrankindices == []

        (dtoc_degree_3_local, dtoc_degree_3_global) =
            Raven.materializedtoc(forest, ghost, nodes, quadrantcommpattern, MPI.COMM_WORLD)
        @test dtoc_degree_3_local == dtoc_degree_3_global
        @test dtoc_degree_3_local == P4estTypes.unsafe_element_nodes(nodes) .+ 0x1

        cell_degree_3 = LobattoCell{Tuple{4,4,4},Float64,Array}()
        dtoc_degree_3 =
            Raven.materializedtoc(cell_degree_3, dtoc_degree_3_local, dtoc_degree_3_global)
        @test isisomorphic(dtoc_degree_3, dtoc_degree_3_global)

        ctod_degree_3 = Raven.materializectod(dtoc_degree_3)
        @test ctod_degree_3 isa AbstractSparseMatrix
        @test istranspose(ctod_degree_3, dtoc_degree_3)

        quadranttolevel = Int8[
            P4estTypes.level.(Iterators.flatten(forest))
            P4estTypes.level.(P4estTypes.ghosts(ghost))
        ]
        parentnodes = Raven.materializeparentnodes(
            cell_degree_3,
            ctod_degree_3,
            quadranttoglobalids,
            quadranttolevel,
        )
        Np = length(cell_degree_3)
        for lid in eachindex(parentnodes)
            pid = parentnodes[lid]
            qlid = cld(lid, Np)
            qpid = cld(pid, Np)

            @test quadranttolevel[qpid] <= quadranttolevel[qlid]
            if quadranttolevel[qlid] == quadranttolevel[qpid]
                @test quadranttoglobalids[qlid] >= quadranttoglobalids[qpid]
            end
            @test dtoc_degree_3[pid] == dtoc_degree_3[lid]
        end

        GC.@preserve nodes begin
            @test Raven.materializequadranttofacecode(nodes) ==
                  P4estTypes.unsafe_face_code(nodes)
        end
    end

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
            cg = coarsegrid(vertices, cells)
            gm = GridManager(cell, cg)
            grid = generate(gm)

            A = continuoustodiscontinuous(grid)
            pts = points(grid)
            rows = rowvals(A)
            vals = nonzeros(A)
            _, n = size(A)

            ncontinuous = 0
            ndiscontinuous = 0
            for j = 1:n
                x = pts[rows[first(nzrange(A, j))]]
                if length(nzrange(A, j)) > 0
                    ncontinuous += 1
                end
                for ii in nzrange(A, j)
                    ndiscontinuous += 1
                    @test pts[rows[ii]] ≈ x
                end
            end
            @test ncontinuous == 2N^2 - N
            @test ndiscontinuous == 2N^2

            pts = Adapt.adapt(Array, pts)
            fm = Adapt.adapt(Array, facemaps(grid))

            @test isapprox(pts[fm.vmapM], pts[fm.vmapP])
            @test isapprox(pts[fm.vmapM[fm.mapM]], pts[fm.vmapM[fm.mapP]])

            bc = boundarycodes(grid)
            vmapM = reshape(fm.vmapM, (N, 4, :))
            for q = 1:numcells(grid)
                for f = 1:4
                    if bc[f, q] == 1
                        @test all(
                            isapprox.(
                                one(FT),
                                map(x -> maximum(abs.(x)), pts[vmapM[:, f, q]]),
                            ),
                        )
                    else
                        @test !all(
                            isapprox.(
                                one(FT),
                                map(x -> maximum(abs.(x)), pts[vmapM[:, f, q]]),
                            ),
                        )
                    end
                end
            end
        end
    end

    let
        cell = LobattoCell{Tuple{2,2,2}}()
        cg = brick((1, 1, 1), (true, true, true))
        gm = GridManager(cell, cg)
        grid = generate(gm)
        fm = facemaps(grid)
        @test fm.vmapP ==
              [2; 4; 6; 8; 1; 3; 5; 7; 3; 4; 7; 8; 1; 2; 5; 6; 5; 6; 7; 8; 1; 2; 3; 4;;]
    end

    let
        cell = LobattoCell{Tuple{2,2}}()
        cg = brick((1, 1), (true, true))
        gm = GridManager(cell, cg)
        grid = generate(gm)
        fm = facemaps(grid)
        @test fm.vmapP == [2; 4; 1; 3; 3; 4; 1; 2;;]
    end

    let
        FT = Float64
        AT = Array

        N = 5
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

            cg = coarsegrid(vertices, cells)
            gm = GridManager(cell, cg)

            grid = generate(gm)

            A = grid.continuoustodiscontinuous
            pts = points(grid)
            rows = rowvals(A)
            vals = nonzeros(A)
            _, n = size(A)

            ncontinuous = 0
            ndiscontinuous = 0
            for j = 1:n
                x = pts[rows[first(nzrange(A, j))]]
                if length(nzrange(A, j)) > 0
                    ncontinuous += 1
                end
                for ii in nzrange(A, j)
                    ndiscontinuous += 1
                    @test pts[rows[ii]] ≈ x
                end
            end
            @test ncontinuous == 2N^3 - N^2
            @test ndiscontinuous == 2N^3

            pts = Adapt.adapt(Array, pts)
            fm = Adapt.adapt(Array, facemaps(grid))

            @test isapprox(pts[fm.vmapM], pts[fm.vmapP])
            @test isapprox(pts[fm.vmapM[fm.mapM]], pts[fm.vmapM[fm.mapP]])

            bc = boundarycodes(grid)
            vmapM = reshape(fm.vmapM, (N, N, 6, :))
            for q = 1:numcells(grid)
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
        end
    end
end
