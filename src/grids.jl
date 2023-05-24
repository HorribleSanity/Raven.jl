abstract type AbstractGrid{C<:AbstractCell} end

floattype(::Type{<:AbstractGrid{C}}) where {C} = floattype(C)
arraytype(::Type{<:AbstractGrid{C}}) where {C} = arraytype(C)
celltype(::Type{<:AbstractGrid{C}}) where {C} = C

floattype(grid::AbstractGrid) = floattype(typeof(grid))
arraytype(grid::AbstractGrid) = arraytype(typeof(grid))
celltype(grid::AbstractGrid) = celltype(typeof(grid))

function WriteVTK.vtk_grid(filename::AbstractString, grid::AbstractGrid, args...; kwargs...)
    return pvtk_grid(filename, grid, args...; kwargs...)
end

function WriteVTK.pvtk_grid(
    filename::AbstractString,
    grid::AbstractGrid,
    args...;
    withghostlayer = Val(false),
    kwargs...,
)
    part = partitionnumber(grid)
    nparts = numberofpartitions(grid)
    vtk = pvtk_grid(
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

struct Grid{C<:AbstractCell,P,L,T,F,PN,N,CTOD,DTOC} <: AbstractGrid{C}
    part::Int
    nparts::Int
    cell::C
    offset::Int
    locallength::Int32
    points::P
    levels::L
    trees::T
    facecodes::F
    parentnodes::PN
    nodecommpattern::N
    continuoustodiscontinuous::CTOD
    discontinuoustocontinuous::DTOC
end

referencecell(grid::Grid) = grid.cell

points(grid::Grid) = points(grid, Val(false))
function points(grid::Grid, withghostlayer::Val{false})
    colons = ntuple(_ -> Colon(), ndims(grid.points) - 1)
    return view(grid.points, colons..., Base.OneTo(grid.locallength))
end
points(grid::Grid, withghostlayer::Val{true}) = grid.points


levels(grid::Grid) = levels(grid, Val(false))
levels(grid::Grid, ::Val{false}) = view(grid.levels, Base.OneTo(grid.locallength))
levels(grid::Grid, ::Val{true}) = grid.levels

trees(grid::Grid) = trees(grid, Val(false))
trees(grid::Grid, ::Val{false}) = view(grid.trees, Base.OneTo(grid.locallength))
trees(grid::Grid, ::Val{true}) = grid.trees

facecodes(grid::Grid) = facecodes(grid, Val(false))
facecodes(grid::Grid, ::Val{false}) = grid.facecodes
facecodes(::Grid, ::Val{true}) =
    throw(error("Face codes are currently stored for local quadrants."))

nodecommpattern(grid::Grid) = grid.nodecommpattern
continuoustodiscontinuous(grid::Grid) = grid.continuoustodiscontinuous

offset(grid::Grid) = grid.offset
numcells(grid::Grid) = numcells(grid, Val(false))
numcells(grid::Grid, ::Val{false}) = grid.locallength
numcells(grid::Grid, ::Val{true}) = length(grid.levels)

Base.length(grid::Grid) = grid.locallength
partitionnumber(grid::Grid) = grid.part
numberofpartitions(grid::Grid) = grid.nparts

function points_vtk(grid::Grid, withghostlayer = Val(false))
    P = toequallyspaced(referencecell(grid))
    x = P * reshape(points(grid, withghostlayer), size(P, 2), :)

    return reinterpret(reshape, eltype(eltype(x)), vec(adapt(Array, x)))
end

function cells_vtk(grid::Grid, withghostlayer = Val(false))
    type = celltype_vtk(referencecell(grid))
    connectivity = connectivity_vtk(referencecell(grid))

    cells = [
        MeshCell(type, e * length(connectivity) .+ connectivity) for
        e = 0:(numcells(grid, withghostlayer)-1)
    ]

    return cells
end

function data_vtk!(vtk, grid::Grid, withghostlayer = Val(false))
    higherorderdegrees = zeros(Int, 3, numcells(grid, withghostlayer))
    ds = [degrees(referencecell(grid))...]
    higherorderdegrees[1:length(ds), :] .= repeat(ds, 1, numcells(grid, withghostlayer))

    vtk["HigherOrderDegrees", VTKCellData()] = higherorderdegrees
    vtk[VTKCellData()] = Dict("HigherOrderDegrees" => "HigherOrderDegrees")

    vtk["PartitionNumber", VTKCellData()] =
        fill(partitionnumber(grid), numcells(grid, withghostlayer))
    vtk["Level", VTKCellData()] = collect(levels(grid, withghostlayer))
    vtk["Tree", VTKCellData()] = collect(trees(grid, withghostlayer))

    return
end
