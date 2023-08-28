using MPI
MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD

if true
    using CUDA
    using CUDA.CUDAKernels

    if CUDA.functional()
        const AT = CuArray
        const local_comm =
            MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
        CUDA.device!(MPI.Comm_rank(local_comm) % length(CUDA.devices()))
        CUDA.allowscalar(false)
    else
        const AT = Array
    end
else
    const AT = Array
end


using Raven
using WriteVTK
using StaticArrays
using Adapt

N = (4, 4, 4)

coarse_grid = coarsegrid("../Snake.inp")
gm = GridManager(LobattoCell{Tuple{N...},Float64,AT}(), coarse_grid, min_level=1)

grid = generate(gm)

vtk_grid("gridcurve", grid) do vtk
    vtk["CellNumber"] = (1:length(grid)) .+ Raven.offset(grid)
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid), size(P, 2), :)
    x = adapt(Array, x)
    vtk["x"] = x
end

