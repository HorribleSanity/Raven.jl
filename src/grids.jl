abstract type AbstractGrid{C<:AbstractCell} end

floattype(::Type{<:AbstractGrid{C}}) where {C} = floattype(C)
arraytype(::Type{<:AbstractGrid{C}}) where {C} = arraytype(C)
celltype(::Type{<:AbstractGrid{C}}) where {C} = C

floattype(grid::AbstractGrid) = floattype(typeof(grid))
arraytype(grid::AbstractGrid) = arraytype(typeof(grid))
celltype(grid::AbstractGrid) = celltype(typeof(grid))

struct Grid{C<:AbstractCell,P,L,T,F,PN,N,CTOD,DTOC} <: AbstractGrid{C}
    comm::MPI.Comm
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

comm(grid::Grid) = grid.comm

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
