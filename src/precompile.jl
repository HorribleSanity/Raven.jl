function precompile_workload(FT, AT)
    lcell2d = LobattoCell{FT,AT}(2, 2)
    vertices2d = [
        SVector{2,FT}(0, 0), # 1
        SVector{2,FT}(2, 0), # 2
        SVector{2,FT}(0, 2), # 3
        SVector{2,FT}(2, 2), # 4
        SVector{2,FT}(4, 0), # 5
        SVector{2,FT}(4, 2), # 6
    ]
    cells2d = [(1, 2, 3, 4), (4, 2, 6, 5)]

    generate(GridManager(lcell2d, brick(FT, 1, 1)))
    generate(GridManager(lcell2d, coarsegrid(vertices2d, cells2d)))

    lcell3d = LobattoCell{FT,AT}(2, 2, 2)
    vertices3d = [
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
    cells3d = [(1, 2, 3, 4, 5, 6, 7, 8), (4, 2, 10, 9, 8, 6, 12, 11)]

    generate(GridManager(lcell3d, brick(FT, 1, 1, 1)))
    generate(GridManager(lcell3d, coarsegrid(vertices3d, cells3d)))

    GaussCell{FT,AT}(2, 2)
    GaussCell{FT,AT}(2, 2, 2)

    return
end
