function recursive_fieldtypes(::Type{T}, ::Type{U} = Real) where {T,U}
    if T <: U
        return (T,)
    else
        return recursive_fieldtypes.(fieldtypes(T), U)
    end
end

@inline function insert(I::NTuple{N,Int}, ::Val{M}, i::Int) where {N,M}
    m = M::Int
    return (I[1:m-1]..., i, I[m:end]...)::NTuple{N + 1,Int}
end

"""
    GridArray{T,N,A,G,F,L,C,D,W} <: AbstractArray{T,N}

`N`-dimensional array of values of type `T` for each grid point using a
struct-of-arrays like format that is GPU friendly.  Type `T` is assumed to be
a hierarchical `struct` that can be `flatten`ed into an `NTuple{L,E<:Real}`.

The backing data array will be of type `A{E}` and will have the fields of `T`
indexed via index `F`.

`GridArray` also stores values for the ghost cells of the grid which are
accessible if `G==true`.
"""
struct GridArray{T,N,A,G,F,L,C,D,W} <: AbstractArray{T,N}
    """MPI.Comm used for communication"""
    comm::C
    """View of the backing data array without the ghost cells"""
    data::D
    """Backing data array with the ghost cells"""
    datawithghosts::W
    """Dimensions of the array without the ghost cells"""
    dims::NTuple{N,Int}
    """Dimensions of the array with the ghost cells"""
    dimswithghosts::NTuple{N,Int}
end

function GridArray{T}(
    ::UndefInitializer,
    ::Type{A},
    dims::NTuple{N,Int},
    dimswithghosts::NTuple{N,Int},
    comm,
    withghosts::Bool,
    fieldindex::Integer,
) where {T,A,N}
    if !(all(dims[1:end-1] .== dimswithghosts[1:end-1]) && dims[end] <= dimswithghosts[end])
        throw(
            DimensionMismatch(
                "dims ($dims) must equal to dimswithghosts ($dimswithghosts) in all but the last dimension where it should be less than",
            ),
        )
    end

    types = flatten(recursive_fieldtypes(T), DataType)

    L = length(types)::Int
    if L == 0
        throw(ArgumentError("Type T has no Real fields"))
    end

    E = first(types)
    if !allequal(types)
        throw(ArgumentError("Type T has different field types: $types"))
    end

    datawithghosts = A{E}(undef, insert(dimswithghosts, Val(fieldindex), L))
    data = view(datawithghosts, (ntuple(_ -> Colon(), Val(N))..., Base.OneTo(dims[end]))...)

    C = typeof(comm)
    D = typeof(data)
    W = typeof(datawithghosts)

    return GridArray{T,N,A,withghosts,fieldindex,L,C,D,W}(
        comm,
        data,
        datawithghosts,
        dims,
        dimswithghosts,
    )
end

"""
    GridArray{T}(undef, grid::Grid)

Create an array containing elements of type `T` for each point in the grid
(including the ghost cells).  The dimensions of the array is
`(size(celltype(grid))..., length(grid))` as the ghost cells are hidden by
default.

The type `T` is assumed to be able to be interpreted into an `NTuple{M,L}`.
Some example types (some using `StructArrays`) are:
- `T = NamedTuple{(:E,:B),Tuple{SVector{3,ComplexF64},SVector{3,ComplexF64}}}`
- `T = NTuple{5,Int64}`
- `T = SVector{5,Int64}`
- `T = ComplexF32`
- `T = Float32`

Instead of using an array-of-struct style storage, a GPU efficient
struct-of-arrays like storage is used.  For example, instead of storing data
like
```julia-repl
julia> T = Tuple{Int,Int};
julia> data = Array{T}(undef, 3, 4, 2); a .= Ref((1,2))
3×4 Matrix{Tuple{Int64, Int64}}:
 (1, 2)  (1, 2)  (1, 2)  (1, 2)
 (1, 2)  (1, 2)  (1, 2)  (1, 2)
 (1, 2)  (1, 2)  (1, 2)  (1, 2)
```
the data would be stored in the order
```julia-repl
julia> permutedims(reinterpret(reshape, Int, data), (2,3,1,4))
3×4×2×2 Array{Int64, 4}:
[:, :, 1, 1] =
 1  1  1  1
 1  1  1  1
 1  1  1  1

[:, :, 2, 1] =
 2  2  2  2
 2  2  2  2
 2  2  2  2

[:, :, 1, 2] =
 1  1  1  1
 1  1  1  1
 1  1  1  1

[:, :, 2, 2] =
 2  2  2  2
 2  2  2  2
 2  2  2  2
```
For a `GridArray` the indices before the ones associated with `T` (the first
two in the example above) are associated with the degrees-of-freedoms of the
cells.  The one after is associated with the number of cells.
"""
function GridArray{T}(::UndefInitializer, grid::Grid) where {T}
    A = arraytype(grid)
    dims = (size(celltype(grid))..., Int(numcells(grid, Val(false))))
    dimswithghosts = (size(celltype(grid))..., Int(numcells(grid, Val(true))))
    F = ndims(celltype(grid)) + 1

    return GridArray{T}(undef, A, dims, dimswithghosts, comm(grid), false, F)
end

GridArray(::UndefInitializer, grid::Grid) = GridArray{Float64}(undef, grid)

function Base.showarg(io::IO, a::GridArray{T,N,A,G,F}, toplevel) where {T,N,A,G,F}
    !toplevel && print(io, "::")
    print(io, "GridArray{", T, ",", N, ",", A, ",", G, ",", F, "}")
    toplevel && print(io, " with data eltype ", eltype(parent(a)))
    return
end

"""
    viewwithghosts(A::GridArray)

Return a `GridArray` with the same data as `A` but with the ghost cells accessible.
"""
@inline function viewwithghosts(a::GridArray{T,N,A,G,F,L,C,D,W}) where {T,N,A,G,F,L,C,D,W}
    GridArray{T,N,A,true,F,L,C,D,W}(
        a.comm,
        a.data,
        a.datawithghosts,
        a.dims,
        a.dimswithghosts,
    )
end

"""
    get_backend(A::GridArray) -> KernelAbstractions.Backend

Returns the `KernelAbstractions.Backend` used to launch kernels interacting
with `A`.
"""
@inline get_backend(::GridArray{T,N,A}) where {T,N,A} = get_backend(A)
"""
    arraytype(A::GridArray) -> DataType

Returns the `DataType` used to store the data, e.g., `Array` or `CuArray`.
"""
@inline arraytype(::GridArray{T,N,A}) where {T,N,A} = A
"""
    showingghosts(A::GridArray) -> Bool

Predicate indicating if the ghost layer is accessible to `A`.
"""
@inline showingghosts(::GridArray{T,N,A,G}) where {T,N,A,G} = G
"""
    fieldindex(A::GridArray{T})

Returns the index used in `A.data` to store the fields of `T`.
"""
@inline fieldindex(::GridArray{T,N,A,G,F}) where {T,N,A,G,F} = F
"""
    fieldslength(A::GridArray{T})

Returns the number of fields used to store `T`.
"""
@inline fieldslength(::GridArray{T,N,A,G,F,L}) where {T,N,A,G,F,L} = L

"""
    comm(A::GridArray) -> MPI.Comm

MPI communicator used by `A`.
"""
@inline comm(a::GridArray) = a.comm

@inline Base.parent(a::GridArray{T,N,A,G}) where {T,N,A,G} =
    ifelse(G, a.datawithghosts, a.data)

"""
    sizewithghosts(A::GridArray)

Return the underlying "parent array" which includes the ghost cells.
"""
@inline parentwithghosts(a::GridArray) = a.datawithghosts

@inline Base.size(a::GridArray{T,N,A,false}) where {T,N,A} = a.dims
@inline Base.size(a::GridArray{T,N,A,true}) where {T,N,A} = a.dimswithghosts

"""
    sizewithghosts(A::GridArray)

Return a tuple containing the dimensions of `A` including the ghost cells.
"""
@inline sizewithghosts(a::GridArray) = a.dimswithghosts

function Base.similar(a::GridArray{S,N,A,G,F}, ::Type{T}, dims::Dims) where {S,N,A,G,F,T}
    GridArray{T}(undef, A, dims, a.dimswithghosts, comm(a), G, F)
end

@inline function Base.getindex(
    a::GridArray{T,N,A,G,F,L},
    I::Vararg{Int,N},
) where {T,N,A,G,F,L}
    @boundscheck Base.checkbounds_indices(Bool, axes(a), I) || Base.throw_boundserror(a, I)
    data = parent(a)
    d = ntuple(
        i -> (@inbounds getindex(data, insert(I, Val(F), i)...)),
        Val(L),
    )::NTuple{L,eltype(data)}
    return unflatten(T, d)
end

@inline function Base.setindex!(
    a::GridArray{T,N,A,G,F,L},
    v,
    I::Vararg{Int,N},
) where {T,N,A,G,F,L}
    @boundscheck Base.checkbounds_indices(Bool, axes(a), I) || Base.throw_boundserror(a, I)
    data = parent(a)
    vt = flatten(convert(T, v)::T)
    @unroll for i = 1:L
        @inbounds setindex!(data, vt[i], insert(I, Val(F), i)...)
    end
end

LinearAlgebra.norm(a::GridArray) = sqrt(MPI.Allreduce(norm(parent(a))^2, +, comm(a)))

@kernel function fill_kernel!(a, x)
    I = @index(Global)
    @inbounds a[I] = x
end

function Base.fill!(a::GridArray{T}, x) where {T}
    fill_kernel!(get_backend(a), 256)(a, convert(T, x)::T, ndrange = length(a))
end

function Adapt.adapt_structure(to, a::GridArray{T,N,A,G,F,L}) where {T,N,A,G,F,L}
    newcomm = Adapt.adapt(to, a.comm)
    newdatawithghosts = Adapt.adapt(to, a.datawithghosts)
    newdata = view(newdatawithghosts, Base.OneTo.(insert(a.dims, Val(F), L))...)

    NA = arraytype(newdatawithghosts)
    NC = typeof(newcomm)
    ND = typeof(newdata)
    NW = typeof(newdatawithghosts)

    GridArray{T,N,NA,G,F,L,NC,ND,NW}(
        newcomm,
        newdata,
        newdatawithghosts,
        a.dims,
        a.dimswithghosts,
    )
end

Base.BroadcastStyle(::Type{<:GridArray}) = Broadcast.ArrayStyle{GridArray}()

cat_gridarrays(t::Broadcast.Broadcasted, rest...) =
    (cat_gridarrays(t.args...)..., cat_gridarrays(rest...)...)
cat_gridarrays(t::GridArray, rest...) = (t, cat_gridarrays(rest...)...)
cat_gridarrays(::Any, rest...) = cat_gridarrays(rest...)
cat_gridarrays() = ()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridArray}},
    ::Type{T},
) where {T}
    dims = length.(axes(bc))

    gridarrays = cat_gridarrays(bc)

    a = first(gridarrays)
    A = arraytype(a)
    G = showingghosts(a)
    F = fieldindex(a)
    elemdims = sizewithghosts(a)[F:end]

    for b in gridarrays
        if A != arraytype(b) ||
           G != showingghosts(b) ||
           F != fieldindex(b) ||
           MPI.Comm_compare(comm(a), comm(b)) != MPI.IDENT ||
           elemdims != sizewithghosts(b)[F:end]
            throw(ArgumentError("Incompatible GridArray arguments in broadcast"))
        end
    end

    return GridArray{T}(undef, A, dims, (dims[1:F-1]..., elemdims...), comm(a), G, F)
end

@kernel function broadcast_kernel!(dest, bc)
    i = @index(Global)
    @inbounds I = CartesianIndices(dest)[i]
    @inbounds dest[I] = bc[I]
end

@inline function Base.copyto!(dest::GridArray, bc::Broadcast.Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bcprime = Broadcast.preprocess(dest, bc)

    broadcast_kernel!(get_backend(dest), 256)(dest, bcprime, ndrange = length(dest))

    return dest
end

@inline function Base.copyto!(dest::GridArray, src::GridArray)
    copyto!(dest.datawithghosts, src.datawithghosts)
    return dest
end

# We follow GPUArrays approach of coping the whole array to the host when
# outputting a GridArray backed by GPU arrays.
convert_to_cpu(xs) = Adapt.adapt_structure(Array, xs)
function Base.print_array(io::IO, X::GridArray{<:Any,0,<:AbstractGPUArray})
    X = convert_to_cpu(X)
    isassigned(X) ? show(io, X[]) : print(io, "#undef")
end
Base.print_array(io::IO, X::GridArray{<:Any,1,<:AbstractGPUArray}) =
    Base.print_matrix(io, convert_to_cpu(X))
Base.print_array(io::IO, X::GridArray{<:Any,2,<:AbstractGPUArray}) =
    Base.print_matrix(io, convert_to_cpu(X))
Base.print_array(io::IO, X::GridArray{<:Any,<:Any,<:AbstractGPUArray}) =
    Base.show_nd(io, convert_to_cpu(X), Base.print_matrix, true)
function Base.show_nd(
    io::IO,
    X::Raven.GridArray{<:Any,<:Any,<:AbstractGPUArray},
    print_matrix::Function,
    show_full::Bool,
)
    Base.show_nd(io, Raven.convert_to_cpu(X), print_matrix, show_full)
end

@inline components(::Type{T}) where {T} = fieldtypes(T)
@inline components(::Type{<:NamedTuple{E,T}}) where {E,T} = NamedTuple{E}(fieldtypes(T))
@inline components(::Type{T}) where {T<:Complex} = NamedTuple{(:re, :im)}(fieldtypes(T))
@inline components(::Type{T}) where {T<:SArray} = fieldtypes(fieldtype(T, 1))
@inline components(::Type{T}) where {T<:Real} = (T,)

@inline function componentoffset(::Type{T}, ::Type{E}, i::Int) where {T<:SArray,E}
    return componentoffset(fieldtype(T, 1), E, i)
end

@inline function componentoffset(::Type{T}, ::Type{E}, i::Int) where {T,E}
    if T <: E
        return 0
    else
        return Int(fieldoffset(T, i) ÷ sizeof(E))
    end
end

@inline function ncomponents(::Type{T}, ::Type{E}) where {T,E}
    return Int(sizeof(T) ÷ sizeof(E))
end

"""
    components(A::GridArray{T})

Splits `A` into a tuple of `GridArray`s where there is one for each component
of `T`.

Note, the data for the components is shared with the original array.

For example, if `A isa GridArray{SVector{3, Float64}}` then a tuple of type
`NTuple{3, GridArray{Float64}}` would be returned.
"""
function components(a::GridArray{T,N,A,G,F}) where {T,N,A,G,F}
    componenttypes = components(T)
    E = eltype(a.datawithghosts)

    c = comm(a)
    dims = size(a)
    dimswithghosts = sizewithghosts(a)

    comps = ntuple(length(componenttypes)) do n
        Tn = componenttypes[n]

        r = (1:ncomponents(Tn, E)) .+ componentoffset(T, E, n)

        datawithghosts =
            view(a.datawithghosts, setindex(axes(a.datawithghosts), r, F)...)
        data = view(a.data, setindex(axes(a.data), r, F)...)

        L = length(r)
        C = typeof(c)
        D = typeof(data)
        W = typeof(datawithghosts)

        GridArray{Tn,N,A,G,F,L,C,D,W}(c, data, datawithghosts, dims, dimswithghosts)
    end

    if T <: Union{NamedTuple,FieldArray}
        comps = NamedTuple{fieldnames(T)}(comps)
    end

    if T <: SVector
        if Size(T) == Size(1)
            comps = NamedTuple{(:x,)}(comps)
        elseif Size(T) == Size(2)
            comps = NamedTuple{(:x, :y)}(comps)
        elseif Size(T) == Size(3)
            comps = NamedTuple{(:x, :y, :z)}(comps)
        elseif Size(T) == Size(4)
            comps = NamedTuple{(:x, :y, :z, :w)}(comps)
        end
    end

    return comps
end
