function extrude_testsuite(AT, FT)
    @testset "extrude" begin
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

        gm = GridManager(c, e)

        grid = generate(gm)

        pts = points(grid)
        fm = facemaps(grid)

        pts = Adapt.adapt(Array, pts)
        fm = Adapt.adapt(Array, fm)

        @test isapprox(pts[fm.vmapM], pts[fm.vmapP])
        @test isapprox(pts[fm.vmapM[fm.mapM]], pts[fm.vmapM[fm.mapP]])
    end
end
