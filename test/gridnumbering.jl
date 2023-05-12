@testset "Grid Numbering" begin
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
    end
end
