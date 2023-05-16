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
        nodes = P4estTypes.lnodes(forest; ghost, degree = 2)
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
        (dtoc_degree_2_local, dtoc_degree_2_global) =
            Raven.materializedtoc(forest, ghost, nodes, quadrantcommpattern, MPI.COMM_WORLD)
        @test dtoc_degree_2_local == dtoc_degree_2_global
        @test dtoc_degree_2_local == P4estTypes.unsafe_element_nodes(nodes) .+ 0x1

        dtoc = Raven.materializedtoc(
            LobattoCell{Tuple{3,3},Float64,Array}(),
            dtoc_degree_2_local,
            dtoc_degree_2_global,
        )
        @test isisomorphic(dtoc, dtoc_degree_2_global)

        nodes_degree_3 = P4estTypes.lnodes(forest; ghost, degree = 3)
        cell_degree_3 = LobattoCell{Tuple{4,4},Float64,Array}()
        dtoc_degree_3 =
            Raven.materializedtoc(cell_degree_3, dtoc_degree_2_local, dtoc_degree_2_global)
        GC.@preserve nodes_degree_3 begin
            dtoc_degree_3_p4est = P4estTypes.unsafe_element_nodes(nodes_degree_3) .+ 0x1
            @test isisomorphic(dtoc_degree_3, dtoc_degree_3_p4est)
        end

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
        nodes = P4estTypes.lnodes(forest; ghost, degree = 2)
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

        (dtoc_degree_2_local, dtoc_degree_2_global) =
            Raven.materializedtoc(forest, ghost, nodes, quadrantcommpattern, MPI.COMM_WORLD)
        @test dtoc_degree_2_local == dtoc_degree_2_global
        @test dtoc_degree_2_local == P4estTypes.unsafe_element_nodes(nodes) .+ 0x1

        cell_degree_2 = LobattoCell{Tuple{3,3,3},Float64,Array}()
        dtoc_degree_2 =
            Raven.materializedtoc(cell_degree_2, dtoc_degree_2_local, dtoc_degree_2_global)
        @test isisomorphic(dtoc_degree_2, dtoc_degree_2_global)

        cell_degree_3 = LobattoCell{Tuple{4,4,4},Float64,Array}()
        nodes_degree_3 = P4estTypes.lnodes(forest; ghost, degree = 3)
        dtoc_degree_3 =
            Raven.materializedtoc(cell_degree_3, dtoc_degree_2_local, dtoc_degree_2_global)
        GC.@preserve nodes_degree_3 begin
            dtoc_degree_3_p4est = P4estTypes.unsafe_element_nodes(nodes_degree_3) .+ 0x1
            @test isisomorphic(dtoc_degree_3, dtoc_degree_3_p4est)
        end

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
end
