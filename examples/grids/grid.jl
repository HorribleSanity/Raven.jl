# export JULIA_LOAD_PATH="${PWD}/../..:${PWD}/../../lib/HarpyCUDA:" 

using Harpy
using MPI
using WriteVTK
using StaticArrays


MPI.Initialized() || MPI.Init()

const comm = MPI.COMM_WORLD

if true
    using HarpyCUDA
    using CUDA
    const AT = CuArray
    const local_comm =
        MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    CUDA.device!(MPI.Comm_rank(local_comm) % length(CUDA.devices()))
    CUDA.allowscalar(false)
else
    AT = Array
end


# here the level is a tuple to 
# coarsegrid = extrude(cubesphere(2, 2), 1)
# gm = GridManager(cell, coarsegrid; level = (2, 3))

N = (3, 2, 5)
K = (2, 3, 4)
coordinates = ntuple(d -> range(start = -1.0, stop = 1.0, length = K[d] + 1), length(K))

gm = GridManager(
    LobattoCell{Float64,AT}(N...),
    Harpy.brick(K...; coordinates);
    comm = comm,
    min_level = 2,
)

indicator = rand((Harpy.AdaptNone, Harpy.AdaptRefine), length(gm))

adapt!(gm, indicator)

warp(x::SVector{2}) = SVector(
    x[1] + cospi(3x[2] / 2) * cospi(x[1] / 2) * cospi(x[2] / 2) / 5,
    x[2] + sinpi(3x[1] / 2) * cospi(x[1] / 2) * cospi(x[2] / 2) / 5,
)

warp(x::SVector{3}) = x

grid = generate(warp, gm)

vtk_grid("grid", grid) do vtk
    vtk["CellNumber"] = (1:length(grid)) .+ offset(grid)
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid), size(P, 2), :)
    vtk["x"] = collect(x)
end

#
## (g, J) = metrics(grid)
## M = mass(grid)
#
## At first we are going use interpolation to avoid the need to geometric
## factors.
#
#x = points(grid)
#f = sin.(norm.(x))
#
#Ac = scatter(grid)
#Acᵀ = gather(grid)
#Φ = globaltolocalinterpolation(grid)
#
#Σ(u) = Φ * Ac * Acᵀ * Φ' * u
