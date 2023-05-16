@testset "Communication" begin
    @test Raven.expand(1:0, 4) == 1:0
    @test Raven.expand([], 4) == []
    @test Raven.expand(3:4, 4) isa UnitRange
    @test Raven.expand(3:4, 4) == 9:16
    @test Raven.expand([3, 4], 4) == 9:16

    let
        recvindices = [7, 8, 9, 10]
        recvranks = Cint[1, 2]
        recvrankindices = [1:1, 2:4]

        sendindices = [1, 3, 1, 3, 5]
        sendranks = Cint[1, 2]
        sendrankindices = [1:2, 3:5]

        pattern = Raven.CommPattern{Array}(
            recvindices,
            recvranks,
            recvrankindices,
            sendindices,
            sendranks,
            sendrankindices,
        )

        pattern = Raven.expand(pattern, 3)

        @test pattern.recvranks == recvranks
        @test pattern.recvindices == [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        @test pattern.recvrankindices == UnitRange{Int64}[1:3, 4:12]
        @test pattern.sendranks == sendranks
        @test pattern.sendindices == [1, 2, 3, 7, 8, 9, 1, 2, 3, 7, 8, 9, 13, 14, 15]
        @test pattern.sendrankindices == UnitRange{Int64}[1:6, 7:15]
    end
end
