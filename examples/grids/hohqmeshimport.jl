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
using HOHQMesh

p = newProject("IceCreamCone", "out")
circ = newCircularArcCurve("outerCircle", [0.0, -1.0, 0.0], 4.0, 0.0, 360.0, "degrees")
addCurveToOuterBoundary!(p, circ)
cone1    = newEndPointsLineCurve("cone1", [0.0, -3.0, 0.0], [1.0, 0.0, 0.0])
iceCream = newCircularArcCurve("iceCream", [0.0, 0.0, 0.0], 1.0, 0.0, 180.0, "degrees")
cone2    = newEndPointsLineCurve("cone2", [-1.0, 0.0, 0.0], [0.0, -3.0, 0.0])
addCurveToInnerBoundary!(p, cone1, "IceCreamCone")
addCurveToInnerBoundary!(p, iceCream, "IceCreamCone")
addCurveToInnerBoundary!(p, cone2, "IceCreamCone")
setPolynomialOrder!(p, 4)
setPlotFileFormat!(p, "sem")
addBackgroundGrid!(p, [0.5, 0.5, 0.0])

# When creating a grid with HOHQMesh.jl the mesh file type must be "ABAQUS" Rather
# than the default format. Such files are .inp. When making the mesh with
# HOHQMesh.jl set the Mesh file type using the following:
#    setMeshFileFormat!(proj::Project,"ABAQUS")

setMeshFileFormat!(p, "ABAQUS")

# this will output the .inp file which we will then import below.
generate_mesh(p)

# import from ABAQUS file
coarse_grid = coarsegrid("out/IceCreamCone.inp")

# Alternatively one could import directly from an existing control file.
# p = openProject("Pond.control", "examples/grids/Pond")
# setMeshFileFormat!(p, "ABAQUS")
# generate_mesh(p)
# N = (3, 3, 3)
# coarse_grid = coarsegrid("examples/grids/Pond/Pond.inp")

N = (4, 4)
gm = GridManager(LobattoCell{Tuple{N...},Float64,AT}(), coarse_grid, min_level=1)

grid = generate(gm)

vtk_grid("IceCreamCone", grid) do vtk
    vtk["CellNumber"] = (1:length(grid)) .+ Raven.offset(grid)
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid), size(P, 2), :)
    x = adapt(Array, x)
    vtk["x"] = x
end
