using CUDA
using CUDA.CUDAKernels
using MPI
using Test
using Raven
using Raven.StaticArrays
using Raven.P4estTypes

MPI.Init()

function isisomorphic(a, b)
    f = Dict{eltype(a),eltype(b)}()
    g = Dict{eltype(b),eltype(a)}()

    for i in eachindex(a, b)
        if a[i] in keys(f)
            if f[a[i]] != b[i]
                return false
            end
        else
            f[a[i]] = b[i]
        end

        if b[i] in keys(g)
            if g[b[i]] != a[i]
                return false
            end
        else
            g[b[i]] = a[i]
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

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    forest = P4estTypes.pxest(Raven.connectivity(cg); comm = MPI.COMM_WORLD)
    P4estTypes.refine!(forest; refine = (_, tid, _) -> tid == 4)
    P4estTypes.balance!(forest)
    ghost = P4estTypes.ghostlayer(forest)
    nodes = P4estTypes.lnodes(forest; ghost, degree = 3)
    P4estTypes.expand!(ghost, forest, nodes)

    forest_self = P4estTypes.pxest(Raven.connectivity(cg); comm = MPI.COMM_SELF)
    P4estTypes.refine!(forest_self; refine = (_, tid, _) -> tid == 4)
    P4estTypes.balance!(forest_self)
    ghost_self = P4estTypes.ghostlayer(forest_self)
    nodes_self = P4estTypes.lnodes(forest_self; ghost = ghost_self, degree = 3)
    P4estTypes.expand!(ghost_self, forest_self, nodes_self)

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

    (dtoc_degree_3_local, dtoc_degree_3_global) =
        Raven.materializedtoc(forest, ghost, nodes, quadrantcommpattern, MPI.COMM_WORLD)

    quadrantcommpattern_self = Raven.materializequadrantcommpattern(forest_self, ghost_self)
    (dtoc_degree_3_local_self, dtoc_degree_3_global_self) = Raven.materializedtoc(
        forest_self,
        ghost_self,
        nodes_self,
        quadrantcommpattern_self,
        MPI.COMM_SELF,
    )

    @test dtoc_degree_3_local == Raven.numbercontiguous(Int32, dtoc_degree_3_global)
    @test eltype(dtoc_degree_3_local) == Int32
    if rank == 0
        @test isisomorphic(
            dtoc_degree_3_global[:, :, 1:1],
            dtoc_degree_3_global_self[:, :, 1:1],
        )
        @test isisomorphic(
            dtoc_degree_3_global[:, :, 2:6],
            dtoc_degree_3_global_self[:, :, 2:6],
        )
    elseif rank == 1
        @test isisomorphic(
            dtoc_degree_3_global[:, :, 1:1],
            dtoc_degree_3_global_self[:, :, 2:2],
        )
        @test isisomorphic(
            dtoc_degree_3_global[:, :, 2:6],
            dtoc_degree_3_global_self[:, :, vcat(1:1, 3:6)],
        )
    elseif rank == 2
        @test isisomorphic(
            dtoc_degree_3_global[:, :, 1:5],
            dtoc_degree_3_global_self[:, :, 3:7],
        )
        @test isisomorphic(
            dtoc_degree_3_global[:, :, 6:7],
            dtoc_degree_3_global_self[:, :, [1, 2]],
        )
    end

    cell_degree_3 = LobattoCell{Float64,Array}(4, 4)
    dtoc_degree_3 =
        Raven.materializedtoc(cell_degree_3, dtoc_degree_3_local, dtoc_degree_3_global)

    if rank == 0
        @test isisomorphic(dtoc_degree_3[:, :, 1:1], dtoc_degree_3_global_self[:, :, 1:1])
        @test isisomorphic(dtoc_degree_3[:, :, 2:6], dtoc_degree_3_global_self[:, :, 2:6])
    elseif rank == 1
        @test isisomorphic(dtoc_degree_3[:, :, 1:1], dtoc_degree_3_global_self[:, :, 2:2])
        @test isisomorphic(
            dtoc_degree_3[:, :, 2:6],
            dtoc_degree_3_global_self[:, :, vcat(1:1, 3:6)],
        )
    elseif rank == 2
        @test isisomorphic(dtoc_degree_3[:, :, 1:5], dtoc_degree_3_global_self[:, :, 3:7])
        @test isisomorphic(
            dtoc_degree_3[:, :, 6:7],
            dtoc_degree_3_global_self[:, :, [1, 2]],
        )
    end

    ctod_degree_3 = Raven.materializectod(dtoc_degree_3)
    nodecommpattern_degree_3 =
        Raven.materializenodecommpattern(cell_degree_3, ctod_degree_3, quadrantcommpattern)
    if rank == 0
        @test nodecommpattern_degree_3.recvindices ==
              [20, 24, 28, 32, 33, 34, 35, 36, 49, 65, 81]
        @test nodecommpattern_degree_3.recvranks == Int32[1, 2]
        @test nodecommpattern_degree_3.recvrankindices == UnitRange{Int64}[1:4, 5:11]
        @test nodecommpattern_degree_3.sendindices == [4, 8, 12, 16, 13, 14, 15, 16]
        @test nodecommpattern_degree_3.sendranks == Int32[1, 2]
        @test nodecommpattern_degree_3.sendrankindices == UnitRange{Int64}[1:4, 5:8]
    elseif rank == 1
        @test nodecommpattern_degree_3.recvindices ==
              [20, 24, 28, 32, 36, 49, 50, 51, 52, 65, 66, 67, 68, 81]
        @test nodecommpattern_degree_3.recvranks == Int32[0, 2]
        @test nodecommpattern_degree_3.recvrankindices == UnitRange{Int64}[1:4, 5:14]
        @test nodecommpattern_degree_3.sendindices == [4, 8, 12, 16, 1, 2, 3, 4]
        @test nodecommpattern_degree_3.sendranks == Int32[0, 2]
        @test nodecommpattern_degree_3.sendrankindices == UnitRange{Int64}[1:4, 5:8]
    elseif rank == 2
        @test nodecommpattern_degree_3.recvindices == [93, 94, 95, 96, 97, 98, 99, 100]
        @test nodecommpattern_degree_3.recvranks == Int32[0, 1]
        @test nodecommpattern_degree_3.recvrankindices == UnitRange{Int64}[1:4, 5:8]
        @test nodecommpattern_degree_3.sendindices ==
              [1, 2, 3, 4, 17, 33, 49, 4, 17, 18, 19, 20, 33, 34, 35, 36, 49]
        @test nodecommpattern_degree_3.sendranks == Int32[0, 1]
        @test nodecommpattern_degree_3.sendrankindices == UnitRange{Int64}[1:7, 8:17]
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

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    forest = P4estTypes.pxest(Raven.connectivity(cg))
    P4estTypes.refine!(forest; refine = (_, tid, _) -> tid == 4)
    P4estTypes.balance!(forest)
    ghost = P4estTypes.ghostlayer(forest)
    nodes = P4estTypes.lnodes(forest; ghost, degree = 3)
    P4estTypes.expand!(ghost, forest, nodes)

    forest_self = P4estTypes.pxest(Raven.connectivity(cg); comm = MPI.COMM_SELF)
    P4estTypes.refine!(forest_self; refine = (_, tid, _) -> tid == 4)
    P4estTypes.balance!(forest_self)
    ghost_self = P4estTypes.ghostlayer(forest_self)
    nodes_self = P4estTypes.lnodes(forest_self; ghost = ghost_self, degree = 3)
    P4estTypes.expand!(ghost_self, forest_self, nodes_self)

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

    (dtoc_degree_3_local, dtoc_degree_3_global) =
        Raven.materializedtoc(forest, ghost, nodes, quadrantcommpattern, MPI.COMM_WORLD)

    quadrantcommpattern_self = Raven.materializequadrantcommpattern(forest_self, ghost_self)
    (dtoc_degree_3_local_self, dtoc_degree_3_global_self) = Raven.materializedtoc(
        forest_self,
        ghost_self,
        nodes_self,
        quadrantcommpattern_self,
        MPI.COMM_SELF,
    )

    @test dtoc_degree_3_local == Raven.numbercontiguous(Int32, dtoc_degree_3_global)
    @test eltype(dtoc_degree_3_local) == Int32
    if rank == 0
        @test isisomorphic(
            dtoc_degree_3_global[:, :, :, 1:1],
            dtoc_degree_3_global_self[:, :, :, 1:1],
        )
        @test isisomorphic(
            dtoc_degree_3_global[:, :, :, 2:9],
            dtoc_degree_3_global_self[:, :, :, [2, 3, 4, 5, 6, 8, 9, 10]],
        )
    elseif rank == 1
        @test isisomorphic(
            dtoc_degree_3_global[:, :, :, 1:1],
            dtoc_degree_3_global_self[:, :, :, 2:2],
        )
        @test isisomorphic(
            dtoc_degree_3_global[:, :, :, 2:9],
            dtoc_degree_3_global_self[:, :, :, [1, 3, 4, 5, 6, 8, 9, 10]],
        )
    elseif rank == 2
        @test isisomorphic(
            dtoc_degree_3_global[:, :, :, 1:9],
            dtoc_degree_3_global_self[:, :, :, 3:11],
        )
        @test isisomorphic(
            dtoc_degree_3_global[:, :, :, 10:11],
            dtoc_degree_3_global_self[:, :, :, [1, 2]],
        )
    end

    cell_degree_3 = LobattoCell{Float64,Array}(4, 4, 4)
    dtoc_degree_3 =
        Raven.materializedtoc(cell_degree_3, dtoc_degree_3_local, dtoc_degree_3_global)
    if rank == 0
        @test isisomorphic(
            dtoc_degree_3[:, :, :, 1:1],
            dtoc_degree_3_global_self[:, :, :, 1:1],
        )
        @test isisomorphic(
            dtoc_degree_3[:, :, :, 2:9],
            dtoc_degree_3_global_self[:, :, :, [2, 3, 4, 5, 6, 8, 9, 10]],
        )
    elseif rank == 1
        @test isisomorphic(
            dtoc_degree_3[:, :, :, 1:1],
            dtoc_degree_3_global_self[:, :, :, 2:2],
        )
        @test isisomorphic(
            dtoc_degree_3[:, :, :, 2:9],
            dtoc_degree_3_global_self[:, :, :, [1, 3, 4, 5, 6, 8, 9, 10]],
        )
    elseif rank == 2
        @test isisomorphic(
            dtoc_degree_3[:, :, :, 1:9],
            dtoc_degree_3_global_self[:, :, :, 3:11],
        )
        @test isisomorphic(
            dtoc_degree_3[:, :, :, 10:11],
            dtoc_degree_3_global_self[:, :, :, [1, 2]],
        )
    end

    ctod_degree_3 = Raven.materializectod(dtoc_degree_3)
    nodecommpattern_degree_3 =
        Raven.materializenodecommpattern(cell_degree_3, ctod_degree_3, quadrantcommpattern)

    dnodes = LinearIndices(dtoc_degree_3)
    if rank == 0
        recv_1 = vec(dnodes[end, :, :, 2])
        recv_2 = vcat(
            vec(dnodes[:, 1, :, 3]),
            dnodes[1, 1, :, 4],
            dnodes[1, 1, :, 5],
            dnodes[1, 1, :, 6],
            dnodes[1, 1, :, 7],
            dnodes[1, 1, :, 8],
            dnodes[1, 1, :, 9],
        )
        send_1 = vec(dnodes[end, :, :, 1])
        send_2 = vec(dnodes[:, end, :, 1])
        sendrecv_ranks = Int32[1, 2]
    elseif rank == 1
        recv_1 = vec(dnodes[end, :, :, 2])
        recv_2 = vcat(
            dnodes[end, 1, :, 3],
            vec(dnodes[:, 1, :, 4]),
            vec(dnodes[:, 1, :, 5]),
            dnodes[1, 1, :, 6],
            vec(dnodes[:, 1, :, 7]),
            vec(dnodes[:, 1, :, 8]),
            dnodes[1, 1, :, 9],
        )
        send_1 = vec(dnodes[end, :, :, 1])
        send_2 = vec(dnodes[:, 1, :, 1])
        sendrecv_ranks = Int32[0, 2]
    elseif rank == 2
        recv_1 = vec(dnodes[:, end, :, 10])
        recv_2 = vec(dnodes[:, 1, :, 11])
        send_1 = vcat(
            vec(dnodes[:, 1, :, 1]),
            dnodes[1, 1, :, 2],
            dnodes[1, 1, :, 3],
            dnodes[1, 1, :, 4],
            dnodes[1, 1, :, 6],
            dnodes[1, 1, :, 7],
            dnodes[1, 1, :, 8],
        )
        send_2 = vcat(
            dnodes[end, 1, :, 1],
            vec(dnodes[:, 1, :, 2]),
            vec(dnodes[:, 1, :, 3]),
            dnodes[1, 1, :, 4],
            vec(dnodes[:, 1, :, 6]),
            vec(dnodes[:, 1, :, 7]),
            dnodes[1, 1, :, 8],
        )
        sendrecv_ranks = Int32[0, 1]
    end
    @test nodecommpattern_degree_3.recvindices == vcat(recv_1, recv_2)
    @test nodecommpattern_degree_3.recvranks == sendrecv_ranks
    @test nodecommpattern_degree_3.recvrankindices ==
          UnitRange{Int64}[1:length(recv_1), (1:length(recv_2)) .+ length(recv_1)]
    @test nodecommpattern_degree_3.sendindices == vcat(send_1, send_2)
    @test nodecommpattern_degree_3.sendranks == sendrecv_ranks
    @test nodecommpattern_degree_3.sendrankindices ==
          UnitRange{Int64}[1:length(send_1), (1:length(send_2)) .+ length(send_1)]
end
