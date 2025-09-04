using MPI
using Test
using Raven
using Raven.Adapt
using Raven.StaticArrays
using Raven.SparseArrays

MPI.Init()

let
    FT = Float32
    AT = Array

    b = brick(FT, 3, 1)
    e = extrude(b, 2)

    @test Raven.coordinates(e) == (zero(FT):3, zero(FT):1, zero(FT):2)

    @test Raven.vertices(e) == [
        SVector{3,FT}(0, 0, 0), # 01
        SVector{3,FT}(0, 0, 1), # 02
        SVector{3,FT}(0, 0, 2), # 03
        SVector{3,FT}(1, 0, 0), # 04
        SVector{3,FT}(1, 0, 1), # 05
        SVector{3,FT}(1, 0, 2), # 06
        SVector{3,FT}(0, 1, 0), # 07
        SVector{3,FT}(0, 1, 1), # 08
        SVector{3,FT}(0, 1, 2), # 09
        SVector{3,FT}(1, 1, 0), # 10
        SVector{3,FT}(1, 1, 1), # 11
        SVector{3,FT}(1, 1, 2), # 12
        SVector{3,FT}(2, 0, 0), # 13
        SVector{3,FT}(2, 0, 1), # 14
        SVector{3,FT}(2, 0, 2), # 15
        SVector{3,FT}(2, 1, 0), # 16
        SVector{3,FT}(2, 1, 1), # 17
        SVector{3,FT}(2, 1, 2), # 18
        SVector{3,FT}(3, 0, 0), # 19
        SVector{3,FT}(3, 0, 1), # 20
        SVector{3,FT}(3, 0, 2), # 21
        SVector{3,FT}(3, 1, 0), # 22
        SVector{3,FT}(3, 1, 1), # 23
        SVector{3,FT}(3, 1, 2), # 24
    ]

    # z = 2
    #   09-----12-----21-----24
    #    |      |      |      |
    #    |      |      |      |
    #    |      |      |      |
    #   03-----06-----15-----18

    # z = 1
    #   08-----11-----20-----23
    #    |      |      |      |
    #    |      |      |      |
    #    |      |      |      |
    #   02-----05-----14-----17

    # z = 0
    #   07-----10-----19-----22
    #    |      |      |      |
    #    |      |      |      |
    #    |      |      |      |
    #   01-----04-----13-----16

    @test Raven.cells(e) == [
        (01, 04, 07, 10, 02, 05, 08, 11)
        (02, 05, 08, 11, 03, 06, 09, 12)
        (04, 13, 10, 16, 05, 14, 11, 17)
        (05, 14, 11, 17, 06, 15, 12, 18)
        (13, 19, 16, 22, 14, 20, 17, 23)
        (14, 20, 17, 23, 15, 21, 18, 24)
    ]

    c = LobattoCell{FT,AT}(2, 2, 2)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    gm = GridManager(c, e; comm)

    grid = generate(gm)
    fm = facemaps(grid)

    ncp = nodecommpattern(grid)
    if rank == 0
        # Grid point numbering aka LinearIndices(points(grid, Val(true)))
        # [:, :, 1, 1] =
        #  1  3
        #  2  4
        #
        # [:, :, 2, 1] =
        #  5  7
        #  6  8
        #
        # [:, :, 1, 2] =
        #   9  11
        #  10  12
        #
        # [:, :, 2, 2] =
        #  13  15
        #  14  16
        #
        # [:, :, 1, 3] =
        #  17  19
        #  18  20
        #
        # [:, :, 2, 3] =
        #  21  23
        #  22  24
        #
        # [:, :, 1, 4] =
        #  25  27
        #  26  28
        #
        # [:, :, 2, 4] =
        #  29  31
        #  30  32

        @test levels(grid, Val(false)) == Int8[0, 0]
        @test levels(grid, Val(true)) == Int8[0, 0, 0, 0]
        @test trees(grid, Val(false)) == Int32[1, 2]
        @test trees(grid, Val(true)) == Int32[1, 2, 3, 4]
        @test points(grid) == SVector{3,FT}[
            (0, 0, 0) (0, 1, 0); (1, 0, 0) (1, 1, 0);;; (0, 0, 1) (0, 1, 1); (1, 0, 1) (1, 1, 1);;;;
            (0, 0, 1) (0, 1, 1); (1, 0, 1) (1, 1, 1);;; (0, 0, 2) (0, 1, 2); (1, 0, 2) (1, 1, 2)
        ]
        @test facecodes(grid, Val(false)) == Int8[0, 0]
        @test boundarycodes(grid) == [1 1; 0 0; 1 1; 1 1; 1 0; 0 1]
        @test ncp.recvranks == Int32[1]
        @test ncp.recvrankindices == UnitRange{Int64}[1:8]
        @test ncp.recvindices == [17, 19, 21, 23, 25, 27, 29, 31]
        @test ncp.sendranks == Int32[1]
        @test ncp.sendrankindices == UnitRange{Int64}[1:8]
        @test ncp.sendindices == [2, 4, 6, 8, 10, 12, 14, 16]
        #! format: off
        @test continuoustodiscontinuous(grid) == sparse(
            [
                1,
                2, 17,
                3,
                4, 19,
                5, 9,
                6, 10, 21, 25,
                7, 11,
                8, 12, 23, 27,
                13,
                14, 29,
                15,
                16, 31,
                18, 20, 22, 24, 26, 28, 30, 32
            ],
            [
                1,
                2, 2,
                3,
                4, 4,
                5, 5,
                6, 6, 6, 6,
                7, 7,
                8, 8, 8, 8,
                9,
                10, 10,
                11,
                12, 12,
                13, 13, 13, 13, 13, 13, 13, 13
            ],
            Bool[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ],
            32,
            13
        )
        #! format: on
        @test communicatingcells(grid) == [1, 2]
        @test noncommunicatingcells(grid) == Int64[]
        @test numcells(grid) == 2
        @test numcells(grid, Val(true)) == 4
        @test offset(grid) == 0
        @test Raven.partitionnumber(grid) == 1
        @test Raven.numberofpartitions(grid) == 3
        @test fm.vmapM == [
            01 09 17 25 #########
            03 11 19 27
            05 13 21 29
            07 15 23 31
            02 10 18 26 #########
            04 12 20 28
            06 14 22 30
            08 16 24 32
            01 09 17 25 #########
            02 10 18 26
            05 13 21 29
            06 14 22 30
            03 11 19 27 #########
            04 12 20 28
            07 15 23 31
            08 16 24 32
            01 09 17 25 #########
            02 10 18 26
            03 11 19 27
            04 12 20 28
            05 13 21 29 #########
            06 14 22 30
            07 15 23 31
            08 16 24 32
        ]
        @test fm.vmapP == [
            01 09 17 25 #########
            03 11 19 27
            05 13 21 29
            07 15 23 31
            17 25 18 26 #########
            19 27 20 28
            21 29 22 30
            23 31 24 32
            01 09 17 25 #########
            02 10 18 26
            05 13 21 29
            06 14 22 30
            03 11 19 27 #########
            04 12 20 28
            07 15 23 31
            08 16 24 32
            01 05 17 25 #########
            02 06 18 26
            03 07 19 27
            04 08 20 28
            09 13 21 29 #########
            10 14 22 30
            11 15 23 31
            12 16 24 32
        ]
    elseif rank == 1
        # Grid point numbering aka LinearIndices(points(grid, Val(true)))
        # [:, :, 1, 1] =
        #  1  3
        #  2  4
        #
        # [:, :, 2, 1] =
        #  5  7
        #  6  8
        #
        # [:, :, 1, 2] =
        #   9  11
        #  10  12
        #
        # [:, :, 2, 2] =
        #  13  15
        #  14  16
        #
        # [:, :, 1, 3] =
        #  17  19
        #  18  20
        #
        # [:, :, 2, 3] =
        #  21  23
        #  22  24
        #
        # [:, :, 1, 4] =
        #  25  27
        #  26  28
        #
        # [:, :, 2, 4] =
        #  29  31
        #  30  32
        #
        # [:, :, 1, 5] =
        #  33  35
        #  34  36
        #
        # [:, :, 2, 5] =
        #  37  39
        #  38  40
        #
        # [:, :, 1, 6] =
        #  41  43
        #  42  44
        #
        # [:, :, 2, 6] =
        #  45  47
        #  46  48

        @test levels(grid, Val(false)) == Int8[0, 0]
        @test levels(grid, Val(true)) == Int8[0, 0, 0, 0, 0, 0]
        @test trees(grid, Val(false)) == Int32[3, 4]
        @test trees(grid, Val(true)) == Int32[3, 4, 1, 2, 5, 6]
        @test points(grid) == SVector{3,FT}[
            (1, 0, 0) (1, 1, 0); (2, 0, 0) (2, 1, 0);;; (1, 0, 1) (1, 1, 1); (2, 0, 1) (2, 1, 1);;;;
            (1, 0, 1) (1, 1, 1); (2, 0, 1) (2, 1, 1);;; (1, 0, 2) (1, 1, 2); (2, 0, 2) (2, 1, 2)
        ]
        @test facecodes(grid, Val(false)) == Int8[0, 0]
        @test boundarycodes(grid) == [0 0; 0 0; 1 1; 1 1; 1 0; 0 1]
        @test ncp.recvranks == Int32[0, 2]
        @test ncp.recvrankindices == UnitRange{Int64}[1:8, 9:16]
        @test ncp.recvindices ==
              [18, 20, 22, 24, 26, 28, 30, 32, 33, 35, 37, 39, 41, 43, 45, 47]
        @test ncp.sendranks == Int32[0, 2]
        @test ncp.sendrankindices == UnitRange{Int64}[1:8, 9:16]
        @test ncp.sendindices == [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16]

        #! format: off
        @test continuoustodiscontinuous(grid) == sparse(
            [
                2, 33,
                4, 35,
                6, 10, 37, 41,
                8, 12, 39, 43,
                14, 45,
                16, 47,
                1, 18,
                3, 20,
                5, 9, 22, 26,
                7, 11, 24, 28,
                13, 30,
                15, 32,
                17, 19, 21, 23, 25, 27, 29, 31, 34, 36, 38, 40, 42, 44, 46, 48
            ],
            [
                1, 1,
                2, 2,
                3, 3, 3, 3,
                4, 4, 4, 4,
                5, 5,
                6, 6,
                7, 7,
                8, 8,
                9, 9, 9, 9,
                10, 10, 10, 10,
                11, 11,
                12, 12,
                13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13
            ],
            Bool[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1
            ],
            48,
            13
        )
        #! format: on
        @test communicatingcells(grid) == [1, 2]
        @test noncommunicatingcells(grid) == Int64[]
        @test numcells(grid) == 2
        @test numcells(grid, Val(true)) == 6
        @test offset(grid) == 2
        @test Raven.partitionnumber(grid) == 2
        @test Raven.numberofpartitions(grid) == 3
        @test fm.vmapM == [
            01 09 17 25 33 41 #########
            03 11 19 27 35 43
            05 13 21 29 37 45
            07 15 23 31 39 47
            02 10 18 26 34 42 #########
            04 12 20 28 36 44
            06 14 22 30 38 46
            08 16 24 32 40 48
            01 09 17 25 33 41 #########
            02 10 18 26 34 42
            05 13 21 29 37 45
            06 14 22 30 38 46
            03 11 19 27 35 43 #########
            04 12 20 28 36 44
            07 15 23 31 39 47
            08 16 24 32 40 48
            01 09 17 25 33 41 #########
            02 10 18 26 34 42
            03 11 19 27 35 43
            04 12 20 28 36 44
            05 13 21 29 37 45 #########
            06 14 22 30 38 46
            07 15 23 31 39 47
            08 16 24 32 40 48
        ]
        @test fm.vmapP == [
            18 26 17 25 33 41 #########
            20 28 19 27 35 43
            22 30 21 29 37 45
            24 32 23 31 39 47
            33 41 18 26 34 42 #########
            35 43 20 28 36 44
            37 45 22 30 38 46
            39 47 24 32 40 48
            01 09 17 25 33 41 #########
            02 10 18 26 34 42
            05 13 21 29 37 45
            06 14 22 30 38 46
            03 11 19 27 35 43 #########
            04 12 20 28 36 44
            07 15 23 31 39 47
            08 16 24 32 40 48
            01 05 17 25 33 41 #########
            02 06 18 26 34 42
            03 07 19 27 35 43
            04 08 20 28 36 44
            09 13 21 29 37 45 #########
            10 14 22 30 38 46
            11 15 23 31 39 47
            12 16 24 32 40 48
        ]
    elseif rank == 2
        # Grid point numbering aka LinearIndices(points(grid, Val(true)))
        # [:, :, 1, 1] =
        #  1  3
        #  2  4
        #
        # [:, :, 2, 1] =
        #  5  7
        #  6  8
        #
        # [:, :, 1, 2] =
        #   9  11
        #  10  12
        #
        # [:, :, 2, 2] =
        #  13  15
        #  14  16
        #
        # [:, :, 1, 3] =
        #  17  19
        #  18  20
        #
        # [:, :, 2, 3] =
        #  21  23
        #  22  24
        #
        # [:, :, 1, 4] =
        #  25  27
        #  26  28
        #
        # [:, :, 2, 4] =
        #  29  31
        #  30  32

        @test levels(grid, Val(false)) == Int8[0, 0]
        @test levels(grid, Val(true)) == Int8[0, 0, 0, 0]
        @test trees(grid, Val(false)) == Int32[5, 6]
        @test trees(grid, Val(true)) == Int32[5, 6, 3, 4]
        @test points(grid) == SVector{3,FT}[
            (2, 0, 0) (2, 1, 0); (3, 0, 0) (3, 1, 0);;; (2, 0, 1) (2, 1, 1); (3, 0, 1) (3, 1, 1);;;;
            (2, 0, 1) (2, 1, 1); (3, 0, 1) (3, 1, 1);;; (2, 0, 2) (2, 1, 2); (3, 0, 2) (3, 1, 2)
        ]
        @test facecodes(grid, Val(false)) == Int8[0, 0]
        @test boundarycodes(grid) == [0 0; 1 1; 1 1; 1 1; 1 0; 0 1]
        @test ncp.recvranks == Int32[1]
        @test ncp.recvrankindices == UnitRange{Int64}[1:8]
        @test ncp.recvindices == [18, 20, 22, 24, 26, 28, 30, 32]
        @test ncp.sendranks == Int32[1]
        @test ncp.sendrankindices == UnitRange{Int64}[1:8]
        @test ncp.sendindices == [1, 3, 5, 7, 9, 11, 13, 15]
        #! format: off
        @test continuoustodiscontinuous(grid) == sparse(
            [
                2,
                4,
                6, 10,
                8, 12,
                14,
                16,
                1, 18,
                3, 20,
                5, 9, 22, 26,
                7, 11, 24, 28,
                13, 30,
                15, 32,
                17, 19, 21, 23, 25, 27, 29, 31
            ],
            [
                1,
                2,
                3, 3,
                4, 4,
                5,
                6,
                7, 7,
                8, 8,
                9, 9, 9, 9,
                10, 10, 10, 10,
                11, 11,
                12, 12,
                13, 13, 13, 13, 13, 13, 13, 13
            ],
            Bool[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ],
            32,
            13
        )
        #! format: on
        @test communicatingcells(grid) == [1, 2]
        @test noncommunicatingcells(grid) == Int64[]
        @test numcells(grid) == 2
        @test numcells(grid, Val(true)) == 4
        @test offset(grid) == 4
        @test Raven.partitionnumber(grid) == 3
        @test Raven.numberofpartitions(grid) == 3
        @test fm.vmapM == [
            01 09 17 25 #########
            03 11 19 27
            05 13 21 29
            07 15 23 31
            02 10 18 26 #########
            04 12 20 28
            06 14 22 30
            08 16 24 32
            01 09 17 25 #########
            02 10 18 26
            05 13 21 29
            06 14 22 30
            03 11 19 27 #########
            04 12 20 28
            07 15 23 31
            08 16 24 32
            01 09 17 25 #########
            02 10 18 26
            03 11 19 27
            04 12 20 28
            05 13 21 29 #########
            06 14 22 30
            07 15 23 31
            08 16 24 32
        ]
        @test fm.vmapP == [
            18 26 17 25 #########
            20 28 19 27
            22 30 21 29
            24 32 23 31
            02 10 18 26 #########
            04 12 20 28
            06 14 22 30
            08 16 24 32
            01 09 17 25 #########
            02 10 18 26
            05 13 21 29
            06 14 22 30
            03 11 19 27 #########
            04 12 20 28
            07 15 23 31
            08 16 24 32
            01 05 17 25 #########
            02 06 18 26
            03 07 19 27
            04 08 20 28
            09 13 21 29 #########
            10 14 22 30
            11 15 23 31
            12 16 24 32
        ]
    end

    # Note that vmapP can point into the ghost layer so we need to get the
    # points including the ghost layer.
    pts = points(grid, Val(true))
    @test isapprox(pts[fm.vmapM[:, 1:numcells(grid)]], pts[fm.vmapP[:, 1:numcells(grid)]])
    @test isapprox(
        pts[fm.vmapM[fm.mapM[:, 1:numcells(grid)]]],
        pts[fm.vmapM[fm.mapP[:, 1:numcells(grid)]]],
    )
end
