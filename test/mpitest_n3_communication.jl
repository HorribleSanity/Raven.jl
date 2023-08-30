using CUDA
using CUDA.CUDAKernels
using MPI
using Test
using Raven

MPI.Init()

let
    mpisize = MPI.Comm_size(MPI.COMM_WORLD)
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)

    @test mpisize == 3

    if mpirank == 0
        datalength = 10

        recvindices = [7, 8, 9, 10]
        recvranks = Cint[1, 2]
        recvrankindices = [1:1, 2:4]

        sendindices = [1, 3, 1, 3, 5]
        sendranks = Cint[1, 2]
        sendrankindices = [1:2, 3:5]

        datacomm = [1, 2, 3, 4, 5, 6, 102, 201, 202, 203]
    elseif mpirank == 1
        datalength = 16

        recvindices = [11, 12, 13, 14, 15, 16]
        recvranks = Cint[0, 2]
        recvrankindices = [1:2, 3:6]

        sendindices = [2, 2, 4, 6, 8, 10]
        sendranks = Cint[0, 2]
        sendrankindices = [1:1, 2:6]

        datacomm =
            [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 1, 3, 202, 203, 204, 205]
    elseif mpirank == 2
        datalength = 20

        recvindices = [8, 9, 10, 14, 15, 16, 17, 18]
        recvranks = Cint[0, 1]
        recvrankindices = [1:3, 4:8]

        sendindices = [1, 2, 3, 2, 3, 4, 5]
        sendranks = Cint[0, 1]
        sendrankindices = [1:3, 4:7]

        datacomm = [
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            1,
            3,
            5,
            211,
            212,
            213,
            102,
            104,
            106,
            108,
            110,
            219,
            220,
        ]
    end

    to2(A) = Float64.(2A)

    data1 = collect((1:datalength) .+ (100mpirank))
    data2 = to2(data1)

    data3 = data1
    data4 = data2

    pattern = Raven.CommPattern{Array}(
        recvindices,
        recvranks,
        recvrankindices,
        sendindices,
        sendranks,
        sendrankindices,
    )

    cm1 = Raven.commmanager(eltype(data1), pattern, Val(false); tag = 1)
    cm2 = Raven.commmanager(eltype(data2), pattern, Val(true); tag = 2)

    Raven.start!(data2, cm2)

    Raven.share!(data1, cm1)

    Raven.finish!(data2, cm2)

    @test data1 == datacomm
    @test data2 == to2(datacomm)

    if CUDA.functional()
        data3 = CuArray(data3)
        data4 = CuArray(data4)

        pattern = Raven.CommPattern{CuArray}(
            CuArray(recvindices),
            recvranks,
            recvrankindices,
            CuArray(sendindices),
            sendranks,
            sendrankindices,
        )

        cm3 = Raven.commmanager(eltype(data3), pattern; tag = 1)
        cm4 = Raven.commmanager(eltype(data4), pattern, Val(true); tag = 2)

        Raven.start!(data4, cm4)

        Raven.share!(data3, cm3)

        Raven.finish!(data4, cm4)

        @test Array(data3) == datacomm
        @test Array(data4) == to2(datacomm)
    end
end
