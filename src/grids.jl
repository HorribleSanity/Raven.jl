abstract type AbstractGrid{C<:AbstractCell} end

floattype(::Type{<:AbstractGrid{C}}) where {C} = floattype(C)
arraytype(::Type{<:AbstractGrid{C}}) where {C} = arraytype(C)
celltype(::Type{<:AbstractGrid{C}}) where {C} = C

floattype(grid::AbstractGrid) = floattype(typeof(grid))
arraytype(grid::AbstractGrid) = arraytype(typeof(grid))
celltype(grid::AbstractGrid) = celltype(typeof(grid))

struct Grid{C<:AbstractCell,P,V,S,L,T,F,B,PN,N,CTOD,DTOC,CC,NCC,FM} <: AbstractGrid{C}
    comm::MPI.Comm
    part::Int
    nparts::Int
    cell::C
    offset::Int
    locallength::Int
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
    noncommunicatingcells::NCC
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

@kernel function min_neighbour_distance_kernel(
    min_neighbour_distance,
    points,
    ::Val{S},
    ::Val{Np},
    ::Val{dims},
) where {S,Np,dims}
    I = @index(Global, Linear)

    @inbounds begin
        e = (I - 1) ÷ Np + 1
        ijk = (I - 1) % Np + 1

        md = typemax(eltype(min_neighbour_distance))
        x⃗ = points[ijk, e]
        for d in dims
            for m in (-1, 1)
                ijknb = ijk + S[d] * m
                if 1 <= ijknb <= Np
                    x⃗nb = points[ijknb, e]
                    md = min(norm(x⃗ - x⃗nb), md)
                end
            end
        end
        min_neighbour_distance[ijk, e] = md
    end
end

function min_node_distance(grid::Grid; dims = 1:ndims(referencecell(grid)))
    A = arraytype(grid)
    T = floattype(grid)
    cell = referencecell(grid)

    if maximum(dims) > ndims(cell) || minimum(dims) < 1
        throw(ArgumentError("dims are not valid"))
    end

    Np = length(cell)
    x = reshape(points(grid), (Np, :))
    min_neighbour_distance = A{T}(undef, size(x))

    min_neighbour_distance_kernel(get_backend(A), 256)(
        min_neighbour_distance,
        x,
        Val(strides(cell)),
        Val(Np),
        Val(dims);
        ndrange = length(grid) * length(cell),
    )

    return MPI.Allreduce(minimum(min_neighbour_distance), min, comm(grid))
end

function faceviews(A::AbstractMatrix, cell::AbstractCell)
    offsets = faceoffsets(cell)
    facesizes = facedims(cell)
    num_faces = length(facesizes)

    if last(offsets) != size(A, 1)
        throw(
            ArgumentError(
                "The first dimension of A needs to contain the face degrees of freedom.",
            ),
        )
    end

    return ntuple(Val(num_faces)) do f
        reshape(view(A, (1+offsets[f]):offsets[f+1], :), facesizes[f]..., :)
    end
end

function Base.show(io::IO, g::Grid)
    compact = get(io, :compact, false)
    print(io, "Raven.Grid{")
    show(io, referencecell(g))
    print(io, "}()")
    if !compact
        nlocal = numcells(g, Val(false))
        print(io, " with $nlocal local elements")
    end

    return
end

function Base.showarg(io::IO, g::Grid, toplevel)
    !toplevel && print(io, "::")

    print(io, "Raven.Grid{")
    Base.showarg(io, referencecell(g), true)
    print(io, "}")

    if toplevel
        nlocal = numcells(g, Val(false))
        print(io, " with $nlocal local elements")
    end

    return
end

Base.summary(io::IO, g::Grid) = Base.showarg(io, g, true)
