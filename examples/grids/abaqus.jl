using MPI
MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD

if false 
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

N = (9, 9)
R = 1

abaqus = Raven.abaqusmeshimport("../GingerbreadMan.inp")
coarse_grid = coarsegrid(abaqus.nodes,abaqus.connectivity) 
gm = GridManager(LobattoCell{Tuple{N...},Float64,AT}(), coarse_grid, min_level = 0)

# ℓ = [0, h] x [0, h]   This is where the points of ga live
# ε = the closed region bounded by its four faces 
#
# T⃗: ℓ -> ε 
#
#                                                     F2(r)
#         *----------*                            *----------*
#         |          |       --T⃗->                 \         |
#         |          |                        F1(s) )       / F3(s)
#         |          |                             /       (    
#      s  |          |                    y       |         \    
#      |  *----------*                    |       *----------*
#      ℓ--r                               ε--x        F4(r)
#       
#    T(r,s) = (1-s)*F1(s) + s*F3(s) + (1-t)*F4(r) + t*F2(r)
#               - (1-s)*(1-t)*F1(0) - (1-s)*t*F1(h) - s*(1-t)*F4(h) - s*t*F3(h)
#    NOTE: F1(0),F1(h),F3(h),F4(h) is in treecoords

grid = generate(gm, abaqus)     #Example of user warp

vtk_grid("gridcurve", grid) do vtk
    vtk["CellNumber"] = (1:length(grid)) .+ Raven.offset(grid)
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid), size(P, 2), :)
    x = adapt(Array,x)
    vtk["x"] = x
end

