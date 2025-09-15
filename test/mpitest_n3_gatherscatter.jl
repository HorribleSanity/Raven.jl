using MPI
using Test
using Raven
using StaticArrays
using LinearAlgebra: norm
import CUDA
import KernelAbstractions as KA


MPI.Init()

function gatherscatterpoints(grid, FT)
    x = points(grid)

    T = eltype(x)

    nans = SVector(ntuple(_ -> FT(NaN), length(eltype(x)))...)

    dones = GridArray{Int}(undef, grid)
    cdegree = GridArray{Int}(undef, grid, Val(true))

    Raven.fillghosts!(dones, -999)
    Raven.fillghosts!(cdegree, -999)

    fill!(dones, one(eltype(dones)))
    gather!(cdegree, dones, grid)

    da = GridArray{T}(undef, grid)
    ca = GridArray{T}(undef, grid, Val(true))

    Raven.fillghosts!(da, nans)
    Raven.fillghosts!(ca, nans)

    fill!(da, nans)
    fill!(ca, nans)

    da .= x
    gather!(ca, da, grid)

    ca ./= cdegree

    fill!(da, nans)
    Raven.fillghosts!(da, nans)
    scatter!(da, ca, grid)

    return da
end

let
    if !MPI.Initialized()
        MPI.Init(threadlevel = :multiple)
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    A = Array
    FT = Float64

    if CUDA.functional() && CUDA.has_cuda_gpu()
        CUDA.allowscalar(false)
        A = CUDA.CuArray
    end

    backend = Raven.get_backend(A)

    if backend isa KA.GPU
        local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
        local_rank = MPI.Comm_rank(local_comm)
        KA.device!(backend, (local_rank % KA.ndevices(backend)) + 1)
    end

    gm = GridManager(LobattoCell{FT,A}(3, 4), brick(3, 2, false, false))
    grid = generate(gm)

    x = points(grid)
    gsx = gatherscatterpoints(grid, FT)

    @test norm(x - gsx) < 10eps(FT)

    gm = GridManager(LobattoCell{FT,A}(3, 4, 5), brick(2, 3, 2, false, false, false))
    grid = generate(gm)

    x = points(grid)
    gsx = gatherscatterpoints(grid, FT)

    @test norm(x - gsx) < 10eps(FT)
end
