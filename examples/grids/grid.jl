# export JULIA_LOAD_PATH="${PWD}/../..:${PWD}/../../lib/HarpyCUDA:" 

using Harpy
using MPI
using WriteVTK


MPI.Initialized() || MPI.Init()

const comm = MPI.COMM_WORLD

if true
    using HarpyCUDA
    using CUDA
    const arraytype = CuArray
    const local_comm =
        MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    CUDA.device!(MPI.Comm_rank(local_comm) % length(CUDA.devices()))
    CUDA.allowscalar(false)
else
    arraytype = Array
end


# here the level is a tuple to 
# coarsegrid = extrude(cubesphere(2, 2), 1)
# gm = GridManager(cell, coarsegrid; level = (2, 3))

gm = GridManager(LobattoCell(3, 4, 2), brick(2, 3, 1); min_level = 2)

indicator = fill(Harpy.AdaptNone, length(gm))
indicator[1] = Harpy.AdaptRefine

adapt!(gm, indicator)

#grid = generate(gm)
#
#vtk_grid("grid", grid) do vtk
#    vtk["CellNumber"] = 1:length(grid)
#end
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
