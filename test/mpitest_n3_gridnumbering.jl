using CUDA
using CUDA.CUDAKernels
using MPI
using Test
using Raven
using Raven.StaticArrays
using Raven.P4estTypes

MPI.Init()

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

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    forest = P4estTypes.pxest(Raven.connectivity(cg); comm = MPI.COMM_WORLD)
    P4estTypes.refine!(forest; refine = (_, tid, _) -> tid == 4)
    P4estTypes.balance!(forest)
    ghost = P4estTypes.ghostlayer(forest)
    nodes = P4estTypes.lnodes(forest; ghost, degree = 2)
    P4estTypes.expand!(ghost, forest, nodes)

    quadranttoglobalids = Raven.materializequadranttoglobalid(forest, ghost)
    if rank == 0
        @test quadranttoglobalids == [1, 2, 3, 4, 5, 6]
    elseif rank == 1
        @test quadranttoglobalids == [2, 1, 3, 4, 5, 6]
    elseif rank == 2
        @test quadranttoglobalids == [3, 4, 5, 6, 7, 1, 2]
    end

    quadrantcommpattern = Raven.materializequadrantcommpattern(forest, ghost)
    if rank == 0
        @test quadrantcommpattern.recvindices == 2:6
        @test quadrantcommpattern.recvranks == [1, 2]
        @test quadrantcommpattern.recvrankindices == [1:1, 2:5]
        @test quadrantcommpattern.sendindices == [1, 1]
        @test quadrantcommpattern.sendranks == [1, 2]
        @test quadrantcommpattern.sendrankindices == [1:1, 2:2]
    elseif rank == 1
        @test quadrantcommpattern.recvindices == 2:6
        @test quadrantcommpattern.recvranks == [0, 2]
        @test quadrantcommpattern.recvrankindices == [1:1, 2:5]
        @test quadrantcommpattern.sendindices == [1, 1]
        @test quadrantcommpattern.sendranks == [0, 2]
        @test quadrantcommpattern.sendrankindices == [1:1, 2:2]
    elseif rank == 2
        @test quadrantcommpattern.recvindices == 6:7
        @test quadrantcommpattern.recvranks == [0, 1]
        @test quadrantcommpattern.recvrankindices == [1:1, 2:2]
        @test quadrantcommpattern.sendindices == [1, 2, 3, 4, 1, 2, 3, 4]
        @test quadrantcommpattern.sendranks == [0, 1]
        @test quadrantcommpattern.sendrankindices == [1:4, 5:8]
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
        (12, 13, 16, 17, 3, 4, 7, 8), # 3
        (4, 6, 8, 9, 13, 15, 17, 18), # 4
    ]
    cg = coarsegrid(vertices, cells)

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    forest = P4estTypes.pxest(Raven.connectivity(cg))
    P4estTypes.refine!(forest; refine = (_, tid, _) -> tid == 4)
    P4estTypes.balance!(forest)
    ghost = P4estTypes.ghostlayer(forest)
    nodes = P4estTypes.lnodes(forest; ghost, degree = 2)
    P4estTypes.expand!(ghost, forest, nodes)

    quadranttoglobalids = Raven.materializequadranttoglobalid(forest, ghost)
    if rank == 0
        @test quadranttoglobalids == [1, 2, 3, 4, 5, 6, 8, 9, 10]
    elseif rank == 1
        @test quadranttoglobalids == [2, 1, 3, 4, 5, 6, 8, 9, 10]
    elseif rank == 2
        @test quadranttoglobalids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
    end

    quadrantcommpattern = Raven.materializequadrantcommpattern(forest, ghost)
    if rank == 0
        @test quadrantcommpattern.recvindices == 2:9
        @test quadrantcommpattern.recvranks == [1, 2]
        @test quadrantcommpattern.recvrankindices == [1:1, 2:8]
        @test quadrantcommpattern.sendindices == [1, 1]
        @test quadrantcommpattern.sendranks == [1, 2]
        @test quadrantcommpattern.sendrankindices == [1:1, 2:2]
    elseif rank == 1
        @test quadrantcommpattern.recvindices == 2:9
        @test quadrantcommpattern.recvranks == [0, 2]
        @test quadrantcommpattern.recvrankindices == [1:1, 2:8]
        @test quadrantcommpattern.sendindices == [1, 1]
        @test quadrantcommpattern.sendranks == [0, 2]
        @test quadrantcommpattern.sendrankindices == [1:1, 2:2]
    elseif rank == 2
        @test quadrantcommpattern.recvindices == 10:11
        @test quadrantcommpattern.recvranks == [0, 1]
        @test quadrantcommpattern.recvrankindices == [1:1, 2:2]
        @test quadrantcommpattern.sendindices == [1, 2, 3, 4, 6, 7, 8, 1, 2, 3, 4, 6, 7, 8]
        @test quadrantcommpattern.sendranks == [0, 1]
        @test quadrantcommpattern.sendrankindices == [1:7, 8:14]
    end
end
