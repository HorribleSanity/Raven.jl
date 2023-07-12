using MPI

MPI.Initialized() || MPI.Init()

const comm = MPI.COMM_WORLD

using Raven
using WriteVTK
using StaticArrays

# here the level is a tuple to 
# coarsegrid = extrude(cubesphere(2, 2), 1)
# gm = GridManager(cell, coarsegrid; level = (2, 3))

N = (3, 3)
R = 1
nedge = 3

coarse_grid = Raven.cubeshellgrid(R, nedge)

gm = GridManager(LobattoCell{Tuple{N...},Float64,Array}(), coarse_grid, min_level = 2)

indicator = rand((Raven.AdaptNone, Raven.AdaptRefine), length(gm))
adapt!(gm, indicator)

function cubespherewarp(point::SVector{3})
    # Put the points in reverse magnitude order
    p = sortperm(abs.(point))
    point = point[p]

    # Convert to angles
    ξ = π * point[2] / 4point[3]
    η = π * point[1] / 4point[3]

    # Compute the ratios
    y_x = tan(ξ)
    z_x = tan(η)

    # Compute the new points
    x = point[3] / hypot(1, y_x, z_x)
    y = x * y_x
    z = x * z_x

    # Compute the new points and unpermute
    point = SVector(z, y, x)[sortperm(p)]

    return point
end

grid = generate(cubespherewarp, gm)

vtk_grid("grid", grid) do vtk
    vtk["CellNumber"] = (1:length(grid)) .+ Raven.offset(grid)
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid), size(P, 2), :)
    vtk["x"] = collect(x)
end
