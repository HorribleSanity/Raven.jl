abstract type AbstractGrid{C<:AbstractCell} end

floattype(::Type{<:AbstractGrid{C}}) where {C} = floattype(C)
arraytype(::Type{<:AbstractGrid{C}}) where {C} = arraytype(C)
celltype(::Type{<:AbstractGrid{C}}) where {C} = C

floattype(grid::AbstractGrid) = floattype(typeof(grid))
arraytype(grid::AbstractGrid) = arraytype(typeof(grid))
celltype(grid::AbstractGrid) = celltype(typeof(grid))

struct Grid{C<:AbstractCell,P,V,S,L,T,F,B,PN,N,CTOD,DTOC,CC,FM} <: AbstractGrid{C}
    comm::MPI.Comm
    part::Int
    nparts::Int
    cell::C
    offset::Int
    locallength::Int32
    points::P
    volumemetrics::V
    surfacemetrics::S
    levels::L
    trees::T
    facecodes::F
    boundarycodes::B
    parentnodes::PN
    nodecommpattern::N
    continuoustodiscontinuous::CTOD
    discontinuoustocontinuous::DTOC
    communicatingcells::CC
    noncommunicatingcells::CC
    facemaps::FM
end

comm(grid::Grid) = grid.comm

referencecell(grid::Grid) = grid.cell

points(grid::Grid) = points(grid, Val(false))
points(grid::Grid, withghostlayer::Val{false}) = viewwithoutghosts(grid.points)
points(grid::Grid, withghostlayer::Val{true}) = viewwithghosts(grid.points)

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

boundarycodes(grid::Grid) = boundarycodes(grid, Val(false))
boundarycodes(grid::Grid, ::Val{false}) = grid.boundarycodes
boundarycodes(::Grid, ::Val{true}) =
    throw(error("Boundary codes are currently stored for local quadrants."))

nodecommpattern(grid::Grid) = grid.nodecommpattern
continuoustodiscontinuous(grid::Grid) = grid.continuoustodiscontinuous

communicatingcells(grid::Grid) = grid.communicatingcells
noncommunicatingcells(grid::Grid) = grid.noncommunicatingcells

volumemetrics(grid::Grid) = grid.volumemetrics
surfacemetrics(grid::Grid) = grid.surfacemetrics

facemaps(grid::Grid) = facemaps(grid, Val(false))
facemaps(grid::Grid, ::Val{false}) = grid.facemaps
facemaps(::Grid, ::Val{true}) =
    throw(error("Face maps are currently stored for local quadrants."))

offset(grid::Grid) = grid.offset
numcells(grid::Grid) = numcells(grid, Val(false))
numcells(grid::Grid, ::Val{false}) = grid.locallength
numcells(grid::Grid, ::Val{true}) = length(grid.levels)

Base.length(grid::Grid) = grid.locallength
partitionnumber(grid::Grid) = grid.part
numberofpartitions(grid::Grid) = grid.nparts
