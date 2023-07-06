module RavenWriteVTKExt

using Adapt
using Raven

isdefined(Base, :get_extension) ? (using WriteVTK) : (using ..WriteVTK)

celltype_vtk(::Raven.LobattoLine) = VTKCellTypes.VTK_LAGRANGE_CURVE
celltype_vtk(::Raven.LobattoQuad) = VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL
celltype_vtk(::Raven.LobattoHex) = VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON

function connectivity_vtk(cell::Raven.LobattoLine)
    L = LinearIndices(size(cell))
    return [
        L[1],      # corners
        L[end],
        L[2:(end-1)]..., # interior
    ]
end

function connectivity_vtk(cell::Raven.LobattoQuad)
    L = LinearIndices(size(cell))
    return [
        L[1, 1], # corners
        L[end, 1],
        L[end, end],
        L[1, end],
        L[2:(end-1), 1]..., # edges
        L[end, 2:(end-1)]...,
        L[2:(end-1), end]...,
        L[1, 2:(end-1)]...,
        L[2:(end-1), 2:(end-1)]..., # interior
    ]
end

function connectivity_vtk(cell::Raven.LobattoHex)
    L = LinearIndices(size(cell))
    return [
        L[1, 1, 1], # corners
        L[end, 1, 1],
        L[end, end, 1],
        L[1, end, 1],
        L[1, 1, end],
        L[end, 1, end],
        L[end, end, end],
        L[1, end, end],
        L[2:(end-1), 1, 1]..., # edges
        L[end, 2:(end-1), 1]...,
        L[2:(end-1), end, 1]...,
        L[1, 2:(end-1), 1]...,
        L[2:(end-1), 1, end]...,
        L[end, 2:(end-1), end]...,
        L[2:(end-1), end, end]...,
        L[1, 2:(end-1), end]...,
        L[1, 1, 2:(end-1)]...,
        L[end, 1, 2:(end-1)]...,
        L[1, end, 2:(end-1)]...,
        L[end, end, 2:(end-1)]...,
        L[1, 2:(end-1), 2:(end-1)]..., # faces
        L[end, 2:(end-1), 2:(end-1)]...,
        L[2:(end-1), 1, 2:(end-1)]...,
        L[2:(end-1), end, 2:(end-1)]...,
        L[2:(end-1), 2:(end-1), 1]...,
        L[2:(end-1), 2:(end-1), end]...,
        L[2:(end-1), 2:(end-1), 2:(end-1)]..., # interior
    ]
end

function points_vtk(grid::Raven.Grid, withghostlayer = Val(false))
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid, withghostlayer), size(P, 2), :)

    return reinterpret(reshape, eltype(eltype(x)), vec(adapt(Array, x)))
end

function cells_vtk(grid::Raven.Grid, withghostlayer = Val(false))
    type = celltype_vtk(referencecell(grid))
    connectivity = connectivity_vtk(referencecell(grid))

    cells = [
        MeshCell(type, e * length(connectivity) .+ connectivity) for
        e = 0:(numcells(grid, withghostlayer)-1)
    ]

    return cells
end

function data_vtk!(vtk, grid::Raven.Grid, withghostlayer = Val(false))
    higherorderdegrees = zeros(Int, 3, numcells(grid, withghostlayer))
    ds = [Raven.degrees(referencecell(grid))...]
    higherorderdegrees[1:length(ds), :] .= repeat(ds, 1, numcells(grid, withghostlayer))

    vtk["HigherOrderDegrees", VTKCellData()] = higherorderdegrees
    vtk[VTKCellData()] = Dict("HigherOrderDegrees" => "HigherOrderDegrees")

    vtk["PartitionNumber", VTKCellData()] =
        fill(Raven.partitionnumber(grid), numcells(grid, withghostlayer))
    vtk["Level", VTKCellData()] = collect(levels(grid, withghostlayer))
    vtk["Tree", VTKCellData()] = collect(trees(grid, withghostlayer))

    return
end

function WriteVTK.vtk_grid(
    filename::AbstractString,
    grid::Raven.AbstractGrid,
    args...;
    kwargs...,
)
    return pvtk_grid(filename, grid, args...; kwargs...)
end

function WriteVTK.pvtk_grid(
    filename::AbstractString,
    grid::Raven.AbstractGrid,
    args...;
    withghostlayer = Val(false),
    kwargs...,
)
    part = Raven.partitionnumber(grid)
    nparts = Raven.numberofpartitions(grid)
    vtk = WriteVTK.pvtk_grid(
        filename,
        points_vtk(grid, withghostlayer),
        cells_vtk(grid, withghostlayer),
        args...;
        part,
        nparts,
        kwargs...,
    )
    data_vtk!(vtk, grid, withghostlayer)

    return vtk
end

end # module RavenWriteVTKExt
