#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using MPI

MPI.Initialized() || MPI.Init()

const comm = MPI.COMM_WORLD

using Raven
using WriteVTK
using StaticArrays

if false
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
        #(2, 5, 4, 6), # 2
        (3, 4, 7, 8), # 3
        (4, 6, 8, 9), # 4
    ]
    cg = coarsegrid(vertices, cells)
    N = (4, 4)
else
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

    N = (4, 4, 4)
end



gm = GridManager(LobattoCell{Tuple{N...},Float64,Array}(), cg, comm = comm)

indicator = fill(Raven.AdaptNone, length(gm))
if MPI.Comm_rank(comm) == MPI.Comm_size(comm) - 0x1
    indicator[end] = Raven.AdaptRefine
end
adapt!(gm, indicator)

grid = generate(gm)

ghost = Raven.P4estTypes.ghostlayer(Raven.forest(gm))
nodes = Raven.P4estTypes.lnodes(Raven.forest(gm); ghost, degree = 3)
Raven.P4estTypes.expand!(ghost, Raven.forest(gm), nodes)

vtk_grid("grid", grid; withghostlayer = Val(true)) do vtk
    vtk["CellNumber"] = (1:numcells(grid, Val(true))) .+ offset(grid)
    vtk["x"] = Raven.points_vtk(grid, Val(true))
    vtk["dnodeid"] = 1:length(points(grid, Val(true)))
    vtk["cnodeid"] = grid.discontinuoustocontinuous
    vtk["parentid"] = grid.parentnodes
    pdtoc = similar(grid.discontinuoustocontinuous)
    fill!(pdtoc, -1)
    GC.@preserve nodes begin
        pdtoc[1:length(Raven.P4estTypes.unsafe_element_nodes(nodes))] .=
            vec(Raven.P4estTypes.unsafe_element_nodes(nodes))
    end
    vtk["cnodeid_p4est"] = pdtoc
end
