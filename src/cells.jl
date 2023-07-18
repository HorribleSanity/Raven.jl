abstract type AbstractCell{S<:Tuple,T,A<:AbstractArray,N} end

floattype(::Type{<:AbstractCell{S,T}}) where {S,T} = T
arraytype(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} = A
Base.ndims(::Type{<:AbstractCell{S,T,A,N}}) where {S,T,A,N} = N
Base.size(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} = size_to_tuple(S)
Base.length(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} = tuple_prod(S)
Base.strides(::Type{<:AbstractCell{T,A,S}}) where {S,T,A} =
    Base.size_to_strides(1, size_to_tuple(S)...)

floattype(cell::AbstractCell) = floattype(typeof(cell))
arraytype(cell::AbstractCell) = arraytype(typeof(cell))
Base.ndims(cell::AbstractCell) = Base.ndims(typeof(cell))
Base.size(cell::AbstractCell) = Base.size(typeof(cell))
Base.length(cell::AbstractCell) = Base.length(typeof(cell))
Base.strides(cell::AbstractCell) = Base.strides(typeof(cell))

function lobattooperators_1d(::Type{T}, M) where {T}
    oldprecision = precision(BigFloat)

    # Increase precision of the type used to compute the 1D operators to help
    # ensure any symmetries.  This is not thread safe so an alternative would
    # be to use ArbNumerics.jl which keeps its precision in the type.
    setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(T))) + 2)))

    points, weights = legendregausslobatto(BigFloat, M)
    derivative = spectralderivative(points)
    equallyspacedpoints = range(-one(BigFloat), stop = one(BigFloat), length = M)
    toequallyspaced = spectralinterpolation(points, equallyspacedpoints)
    tolowerhalf = spectralinterpolation(points, (points .- 1) ./ 2)
    toupperhalf = spectralinterpolation(points, (points .+ 1) ./ 2)

    setprecision(oldprecision)

    points = Array{T}(points)
    weights = Array{T}(weights)
    derivative = Array{T}(derivative)
    toequallyspaced = Array{T}(toequallyspaced)
    tohalves = (Array{T}(tolowerhalf), Array{T}(toupperhalf))

    return (; points, weights, derivative, toequallyspaced, tohalves)
end

struct LobattoCell{S,T,A,N,O,P,D,M,FM,E,H,C} <: AbstractCell{S,T,A,N}
    points_1d::O
    weights_1d::O
    points::P
    derivatives::D
    mass::M
    facemass::FM
    toequallyspaced::E
    tohalves_1d::H
    connectivity::C
end

function Base.show(io::IO, ::LobattoCell{S,T,A}) where {S,T,A}
    print(io, "LobattoCell{")
    Base.show(io, S)
    print(io, ", ")
    Base.show(io, T)
    print(io, ", ")
    Base.show(io, A)
    print(io, "}")
end

function LobattoCell{Tuple{S1},T,A}() where {S1,T,A}
    o = adapt(A, lobattooperators_1d(T, S1))
    points_1d = (o.points,)
    weights_1d = (o.weights,)

    points = vec(SVector.(points_1d...))
    derivatives = (Kron((o.derivative,)),)
    mass = Diagonal(vec(.*(weights_1d...)))
    facemass = adapt(A, Diagonal([T(1), T(1)]))
    toequallyspaced = Kron((o.toequallyspaced,))
    tohalves_1d = ((o.tohalves[1], o.tohalves[2]),)
    connectivity = materializeconnectivity(LobattoCell{Tuple{S1},T,A})

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
        connectivity,
    )
    LobattoCell{Tuple{S1},T,A,1,typeof.(args[2:end])...}(args...)
end

function LobattoCell{Tuple{S1,S2},T,A}() where {S1,S2,T,A}
    o = adapt(A, (lobattooperators_1d(T, S1), lobattooperators_1d(T, S2)))

    points_1d = (reshape(o[1].points, (S1, 1)), reshape(o[2].points, (1, S2)))
    weights_1d = (reshape(o[1].weights, (S1, 1)), reshape(o[2].weights, (1, S2)))
    points = vec(SVector.(points_1d...))
    derivatives =
        (Kron((Eye{T,S2}(), o[1].derivative)), Kron((o[2].derivative, Eye{T,S1}())))
    mass = Diagonal(vec(.*(weights_1d...)))
    ω1, ω2 = weights_1d
    facemass = Diagonal(vcat(repeat(vec(ω2), 2), repeat(vec(ω1), 2)))
    toequallyspaced = Kron((o[2].toequallyspaced, o[1].toequallyspaced))
    tohalves_1d =
        ((o[1].tohalves[1], o[1].tohalves[2]), (o[2].tohalves[1], o[2].tohalves[2]))
    connectivity = materializeconnectivity(LobattoCell{Tuple{S1,S2},T,A})

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
        connectivity,
    )
    LobattoCell{Tuple{S1,S2},T,A,2,typeof.(args[2:end])...}(args...)
end

function LobattoCell{Tuple{S1,S2,S3},T,A}() where {S1,S2,S3,T,A}
    o = adapt(
        A,
        (
            lobattooperators_1d(T, S1),
            lobattooperators_1d(T, S2),
            lobattooperators_1d(T, S3),
        ),
    )

    points_1d = (
        reshape(o[1].points, (S1, 1, 1)),
        reshape(o[2].points, (1, S2, 1)),
        reshape(o[3].points, (1, 1, S3)),
    )
    weights_1d = (
        reshape(o[1].weights, (S1, 1, 1)),
        reshape(o[2].weights, (1, S2, 1)),
        reshape(o[3].weights, (1, 1, S3)),
    )
    points = vec(SVector.(points_1d...))
    derivatives = (
        Kron((Eye{T,S3}(), Eye{T,S2}(), o[1].derivative)),
        Kron((Eye{T,S3}(), o[2].derivative, Eye{T,S1}())),
        Kron((o[3].derivative, Eye{T,S2}(), Eye{T,S1}())),
    )
    mass = Diagonal(vec(.*(weights_1d...)))
    ω1, ω2, ω3 = weights_1d
    facemass = Diagonal(
        vcat(repeat(vec(ω2 .* ω3), 2), repeat(vec(ω1 .* ω3), 2), repeat(vec(ω1 .* ω2), 2)),
    )
    toequallyspaced =
        Kron((o[3].toequallyspaced, o[2].toequallyspaced, o[1].toequallyspaced))
    tohalves_1d = (
        (o[1].tohalves[1], o[1].tohalves[2]),
        (o[2].tohalves[1], o[2].tohalves[2]),
        (o[3].tohalves[1], o[3].tohalves[2]),
    )
    connectivity = materializeconnectivity(LobattoCell{Tuple{S1,S2,S3},T,A})

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
        connectivity,
    )
    LobattoCell{Tuple{S1,S2,S3},T,A,3,typeof.(args[2:end])...}(args...)
end

LobattoCell{S,T}() where {S,T} = LobattoCell{S,T,Array}()
LobattoCell{S}() where {S} = LobattoCell{S,Float64}()

function Adapt.adapt_structure(to, cell::LobattoCell{S,T,A,N}) where {S,T,A,N}
    names = fieldnames(LobattoCell)
    args = ntuple(j -> adapt(to, getfield(cell, names[j])), length(names))
    B = arraytype(to)

    LobattoCell{S,T,B,N,typeof.(args[2:end])...}(args...)
end

const LobattoLine{T,A} = LobattoCell{Tuple{B},T,A} where {B}
const LobattoQuad{T,A} = LobattoCell{Tuple{B,C},T,A} where {B,C}
const LobattoHex{T,A} = LobattoCell{Tuple{B,C,D},T,A} where {B,C,D}

points_1d(cell::LobattoCell) = cell.points_1d
weights_1d(cell::LobattoCell) = cell.weights_1d
points(cell::LobattoCell) = cell.points
derivatives(cell::LobattoCell) = cell.derivatives
function derivatives_1d(cell::LobattoCell)
    N = ndims(cell)
    ntuple(i -> cell.derivatives[i].args[N-i+1], Val(N))
end
mass(cell::LobattoCell) = cell.mass
facemass(cell::LobattoCell) = cell.facemass
toequallyspaced(cell::LobattoCell) = cell.toequallyspaced
tohalves_1d(cell::LobattoCell) = cell.tohalves_1d
connectivity(cell::LobattoCell) = cell.connectivity
degrees(cell::LobattoCell) = size(cell) .- 1

function materializeconnectivity(::Type{LobattoCell{Tuple{L},T,A}}) where {L,T,A}
    indices = collect(LinearIndices((L,)))

    conn = (
        (A(indices),), # edge
        ( # corners
            indices[begin],
            indices[end],
        ),
    )

    return conn
end

function materializeconnectivity(::Type{LobattoCell{Tuple{L,M},T,A}}) where {L,M,T,A}
    indices = collect(LinearIndices((L, M)))

    conn = (
        (A(indices),), # face
        ( # edges
            A(indices[begin, begin:end]),
            A(indices[end, begin:end]),
            A(indices[begin:end, begin]),
            A(indices[begin:end, end]),
        ),
        ( # corners
            (indices[begin, begin]),
            (indices[end, begin]),
            (indices[begin, end]),
            (indices[end, end]),
        ),
    )

    return conn
end

function materializeconnectivity(::Type{LobattoCell{Tuple{L,M,N},T,A}}) where {L,M,N,T,A}
    indices = collect(LinearIndices((L, M, N)))

    conn = (
        (A(indices),), # volume
        ( # faces
            A(indices[begin, begin:end, begin:end]),
            A(indices[end, begin:end, begin:end]),
            A(indices[begin:end, begin, begin:end]),
            A(indices[begin:end, end, begin:end]),
            A(indices[begin:end, begin:end, begin]),
            A(indices[begin:end, begin:end, end]),
        ),
        ( # edges
            A(indices[begin, begin, begin:end]),
            A(indices[end, begin, begin:end]),
            A(indices[begin, end, begin:end]),
            A(indices[end, end, begin:end]),
            A(indices[begin, begin:end, begin]),
            A(indices[end, begin:end, begin]),
            A(indices[begin, begin:end, end]),
            A(indices[end, begin:end, end]),
            A(indices[begin:end, begin, begin]),
            A(indices[begin:end, end, begin]),
            A(indices[begin:end, begin, end]),
            A(indices[begin:end, end, end]),
        ),
        ( # corners
            (indices[begin, begin, begin]),
            (indices[end, begin, begin]),
            (indices[begin, end, begin]),
            (indices[end, end, begin]),
            (indices[begin, begin, end]),
            (indices[end, begin, end]),
            (indices[begin, end, end]),
            (indices[end, end, end]),
        ),
    )

    return conn
end

materializefaces(cell::AbstractCell) = materializefaces(typeof(cell))
function materializefaces(::Type{<:LobattoLine})
    return (
        SA[1; 2], # edge
        SA[1 2], # corners
    )
end

function materializefaces(::Type{<:LobattoQuad})
    return (
        SA[1; 2; 3; 4; 5; 6; 7; 8], # face
        SA[
            1 2 1 3
            3 4 2 4
        ], # edges
        SA[1 2 3 4], # corners
    )
end

function materializefaces(::Type{<:LobattoHex})
    return (
        SA[1; 2; 3; 4; 5; 6; 7; 8], # volume
        SA[1 2 1 3 1 5; 3 4 2 4 2 6; 5 6 5 7 3 7; 7 8 6 8 4 8], # faces
        SA[
            1 3 5 7 1 2 5 6 1 2 3 4
            2 4 6 8 3 4 7 8 5 6 7 8
        ], # edges
        SA[1 2 3 4 5 6 7 8], # corners
    )
end

@inline function connectivityoffsets(cell::LobattoCell, ::Val{N}) where {N}
    connectivityoffsets(typeof(cell), Val(N))
end
@inline function connectivityoffsets(::Type{C}, ::Val{1}) where {C<:LobattoLine}
    L, = size(C)
    return (0, L)
end
@inline function connectivityoffsets(::Type{C}, ::Val{2}) where {C<:LobattoLine}
    return (0, 1, 2)
end

@inline function connectivityoffsets(::Type{C}, ::Val{1}) where {C<:LobattoQuad}
    L, M = size(C)
    return (0, L * M)
end
@inline function connectivityoffsets(::Type{C}, ::Val{2}) where {C<:LobattoQuad}
    L, M = size(C)
    return (0, M, 2M, 2M + L, 2M + 2L)
end
@inline function connectivityoffsets(::Type{C}, ::Val{3}) where {C<:LobattoQuad}
    return (0, 1, 2, 3, 4)
end

@inline function connectivityoffsets(::Type{C}, ::Val{1}) where {C<:LobattoHex}
    L, M, N = size(C)
    return (0, L * M * N)
end
@inline function connectivityoffsets(::Type{C}, ::Val{2}) where {C<:LobattoHex}
    L, M, N = size(C)
    return cumsum((0, M * N, M * N, L * N, L * N, L * M, L * M))
end
@inline function connectivityoffsets(::Type{C}, ::Val{3}) where {C<:LobattoHex}
    L, M, N = size(C)
    return cumsum((0, N, N, N, N, M, M, M, M, L, L, L, L))
end
@inline function connectivityoffsets(::Type{C}, ::Val{4}) where {C<:LobattoHex}
    return (0, 1, 2, 3, 4, 5, 6, 7, 8)
end

@kernel function quadpoints!(
    points,
    ri,
    si,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{Q},
) where {I,J,Q}
    i, j, q1 = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, Q)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && q1 == 1
                rl[i] = ri[i]
            end

            if i == 1 && q1 == 1
                sl[j] = si[j]
            end

            if i ≤ 2 && j ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                treecoords[i, j, q1] = coarsegridvertices[vids[i+2*(j-1)]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = ix / P4EST_ROOT_LEN
            cs = iy / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = cr + h * (rl[i] + 1)
            s = cs + h * (sl[j] + 1)

            w1 = (1 - r) * (1 - s)
            w2 = r * (1 - s)
            w3 = (1 - r) * s
            w4 = r * s

            c1 = treecoords[1, 1, q1]
            c2 = treecoords[2, 1, q1]
            c3 = treecoords[1, 2, q1]
            c4 = treecoords[2, 2, q1]

            points[i, j, q] = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4
        end
    end
end

function materializepoints(
    referencecell::LobattoQuad,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    forest,
    comm,
)
    r = vec.(points_1d(referencecell))
    Q = max(512 ÷ prod(length.(r)), 1)

    IntType = typeof(length(r))
    num_local = IntType(P4estTypes.lengthoflocalquadrants(forest))
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)

    kernel! = quadpoints!(backend, (length.(r)..., Q))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...,
        Val(Q);
        ndrange = size(points),
    )

    return points
end

@kernel function hexpoints!(
    points,
    ri,
    si,
    ti,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{K},
) where {I,J,K}
    i, j, k = @index(Local, NTuple)
    q = @index(Group, Linear)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, 2)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)
    tl = @localmem eltype(ti) (K,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && k == 1
                rl[i] = ri[i]
            end

            if i == 1 && k == 1
                sl[j] = si[j]
            end

            if i == 1 && j == 1
                tl[k] = ti[k]
            end

            if i ≤ 2 && j ≤ 2 && k ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                id = i + 2 * (j - 1) + 4 * (k - 1)
                treecoords[i, j, k] = coarsegridvertices[vids[id]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]
            iz = quadranttocoordinate[q, 3]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = ix / P4EST_ROOT_LEN
            cs = iy / P4EST_ROOT_LEN
            ct = iz / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = cr + h * (rl[i] + 1)
            s = cs + h * (sl[j] + 1)
            t = ct + h * (tl[k] + 1)

            w1 = (1 - r) * (1 - s) * (1 - t)
            w2 = r * (1 - s) * (1 - t)
            w3 = (1 - r) * s * (1 - t)
            w4 = r * s * (1 - t)
            w5 = (1 - r) * (1 - s) * t
            w6 = r * (1 - s) * t
            w7 = (1 - r) * s * t
            w8 = r * s * t

            points[i, j, k, q] =
                w1 * treecoords[1] +
                w2 * treecoords[2] +
                w3 * treecoords[3] +
                w4 * treecoords[4] +
                w5 * treecoords[5] +
                w6 * treecoords[6] +
                w7 * treecoords[7] +
                w8 * treecoords[8]
        end
    end
end

function materializepoints(
    referencecell::LobattoHex,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    forest,
    comm,
)
    r = vec.(points_1d(referencecell))

    IntType = typeof(length(r))
    num_local = IntType(P4estTypes.lengthoflocalquadrants(forest))
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)
    kernel! = hexpoints!(backend, length.(r))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...;
        ndrange = size(points),
    )

    return points
end

function _getdims(cellsize, dtoc_degree2_global, node, quad)
    dims = ntuple(length(cellsize)) do n
        # We use a StepRange here so that the return type is the same
        # whether or not the dim gets reversed.
        dim =
            node[n] == 2 ? StepRange(2, Int8(1), cellsize[n] - 1) :
            node[n] == 1 ? StepRange(1, Int8(1), 1) :
            StepRange(cellsize[n], Int8(1), cellsize[n])

        shift = ntuple(m -> m == n ? 1 : 0, length(cellsize))

        # Flip the dimension to match the orientation of the degree 2 node numbering
        if node[n] == 2 &&
           dtoc_degree2_global[(node .+ shift)..., quad] <
           dtoc_degree2_global[(node .- shift)..., quad]
            dim = reverse(dim)
        end

        return dim
    end

    return dims
end

function materializedtoc(cell::LobattoCell, dtoc_degree2_local, dtoc_degree2_global)
    cellsize = size(cell)

    # Compute the offsets for the cell node numbering
    offsets = zeros(Int, maximum(dtoc_degree2_local) + 1)
    for i in eachindex(IndexCartesian(), dtoc_degree2_local)
        l = dtoc_degree2_local[i]
        I = Tuple(i)
        node = I[1:end-1]
        # compute the cell dofs for the corner, edge, face or volume identified by node.
        # This is an exclusive count, so the number of dofs in the volume do not include
        # the ones that are also on the faces, edges, or corners.
        offsets[l+1] = prod(ntuple(n -> node[n] == 2 ? (cellsize[n] - 2) : 1, length(node)))
    end
    cumsum!(offsets, offsets)

    dtoc = zeros(Int, cellsize..., last(size(dtoc_degree2_local)))
    for i in eachindex(IndexCartesian(), dtoc_degree2_local)
        l = dtoc_degree2_local[i]
        I = Tuple(i)
        node = I[1:end-1]
        quad = I[end]

        dims = _getdims(cellsize, dtoc_degree2_global, node, quad)
        for (j, k) in enumerate(CartesianIndices(dims))
            dtoc[k, quad] = j + offsets[l]
        end
    end

    return dtoc
end

function materializenodecommpattern(cell::LobattoCell, ctod, quadrantcommpattern)
    ghostranktompirank = quadrantcommpattern.recvranks
    ghostranktoindices =
        expand.(
            [
                quadrantcommpattern.recvindices[ids] for
                ids in quadrantcommpattern.recvrankindices
            ],
            length(cell),
        )

    ranktype = eltype(ghostranktompirank)
    indicestype = eltype(eltype(ghostranktoindices))

    if length(ghostranktompirank) == 0
        return CommPattern{Array}(
            indicestype[],
            ranktype[],
            UnitRange{indicestype}[],
            indicestype[],
            ranktype[],
            UnitRange{indicestype}[],
        )

    end

    dofstarts = zeros(indicestype, length(ghostranktoindices) + 1)
    for (i, ids) in enumerate(ghostranktoindices)
        dofstarts[i] = first(ids)
    end
    dofstarts[end] = last(last(ghostranktoindices)) + 0x1

    senddofs = Dict{ranktype,Set{indicestype}}()
    recvdofs = Dict{ranktype,Set{indicestype}}()

    rows = rowvals(ctod)
    n = size(ctod, 2)
    remoteranks = Set{ranktype}()
    for j = 1:n
        containslocal = false
        for k in nzrange(ctod, j)
            i = rows[k]
            s = searchsorted(dofstarts, i)
            if last(s) > 0
                push!(remoteranks, ghostranktompirank[last(s)])
            end
            if last(s) == 0
                containslocal = true
            end
        end

        if !isempty(remoteranks)
            for k in nzrange(ctod, j)
                i = rows[k]
                s = searchsorted(dofstarts, i)
                if last(s) == 0
                    for rank in remoteranks
                        # local node we need to send
                        sendset = get!(senddofs, rank) do
                            Set{indicestype}()
                        end
                        push!(sendset, i)
                    end
                elseif containslocal
                    # remote node we need to recv
                    rank = ghostranktompirank[last(s)]
                    recvset = get!(recvdofs, rank) do
                        Set{indicestype}()
                    end
                    push!(recvset, i)
                end
            end
        end

        empty!(remoteranks)
    end

    numsendindices = 0
    for dofs in keys(senddofs)
        numsendindices += length(dofs)
    end

    sendindices = Int[]
    sendrankindices = UnitRange{Int}[]
    sendoffset = 0
    for r in ghostranktompirank
        dofs = senddofs[r]
        append!(sendindices, sort(collect(dofs)))
        push!(sendrankindices, (1:length(dofs)) .+ sendoffset)

        sendoffset += length(dofs)
    end

    recvindices = Int[]
    recvrankindices = UnitRange{Int}[]
    recvoffset = 0
    for r in ghostranktompirank
        dofs = recvdofs[r]
        append!(recvindices, sort(collect(dofs)))
        push!(recvrankindices, (1:length(dofs)) .+ recvoffset)

        recvoffset += length(dofs)
    end

    return CommPattern{Array}(
        recvindices,
        ghostranktompirank,
        recvrankindices,
        sendindices,
        ghostranktompirank,
        sendrankindices,
    )
end

function materializeparentnodes(
    cell::LobattoCell,
    ctod,
    quadranttoglobalid,
    quadranttolevel,
)
    Np = length(cell)
    rows = rowvals(ctod)
    m, n = size(ctod)
    parentdofs = zeros(eltype(rows), m)
    for j = 1:n
        level = typemax(Int8)
        gid = typemax(eltype(quadranttoglobalid))
        pdof = 0
        for ii in nzrange(ctod, j)
            i = rows[ii]
            e = cld(i, Np)
            if quadranttolevel[e] ≤ level && quadranttoglobalid[e] < gid
                level = quadranttolevel[e]
                gid = quadranttoglobalid[e]
                pdof = i
            end
        end
        @assert pdof != 0
        for ii in nzrange(ctod, j)
            i = rows[ii]
            parentdofs[i] = pdof
        end
    end

    return reshape(parentdofs, size(cell)..., :)
end

#function materializepoints(referencecell::LobattoLine, vertices, connectivity)
#    T = floattype(referencecell)
#    A = arraytype(referencecell)
#    r = points_1d(referencecell)
#    p = fieldarray(undef, SVector{1, T}, A,
#                   (size(referencecell)..., length(connectivity)))
#    connectivity = vec(connectivity)
#    vertices = vec(vertices)
#
#    @tullio p[i, e] =
#        @inbounds(begin
#                      c1, c2 = connectivity[e]
#                      ri = $(r[1])[i]
#
#                      #((1 - ri) * vertices[c1] +
#                      # (1 + ri) * vertices[c2]) / 2
#
#                      # this is the same as above but
#                      # explicit muladds help with generating symmetric meshes
#                      (muladd(-ri, vertices[c1], vertices[c1]) +
#                       muladd( ri, vertices[c2], vertices[c2])) / 2
#                  end) (i in axes(p, 1), e in axes(p, 2))
#
#    return reshape(p, (length(referencecell), length(connectivity)))
#end
#
#function materializemetrics(referencecell::LobattoLine, points, unwarpedbrick)
#    T = floattype(referencecell)
#    A = arraytype(referencecell)
#    num_cellindices =  length(referencecell)
#    num_cells = last(size(points))
#    faceconn = connectivity(referencecell)[2]
#    num_facesindices = sum(length.(faceconn))
#
#    metrics = fieldarray(undef, (g=SMatrix{1, 1, T, 1}, J=T), A,
#                         (num_cellindices, num_cells))
#    g, J = components(metrics)
#
#    D₁ = only(derivatives(referencecell))
#    x₁ = only(components(points))
#
#    if unwarpedbrick
#        @tullio avx=false J[i, e] = (x₁[end, e] - x₁[1, e]) / 2
#    else
#        J .= D₁ * x₁
#    end
#    @. g = tuple(inv(J))
#
#    facemetrics = fieldarray(undef, (n=SVector{1, T}, J=T), A,
#                             (num_facesindices, num_cells))
#    n, fJ = components(facemetrics)
#
#    n₁, n₂ = faceviews(referencecell, n)
#
#    @tullio n₁[e] = tuple(-sign(J[  1, e]))
#    @tullio n₂[e] = tuple( sign(J[end, e]))
#    fJ .= oneunit(T)
#
#    return (metrics, facemetrics)
#end
#
#function materializepoints(referencecell::LobattoQuad,
#        vertices::AbstractArray{<:SVector{PDIM}}, connectivity) where PDIM
#    T = floattype(referencecell)
#    A = arraytype(referencecell)
#    r = vec.(points_1d(referencecell))
#
#    p = fieldarray(undef, SVector{PDIM, T}, A,
#                   (size(referencecell)..., length(connectivity)))
#
#    connectivity = vec(connectivity)
#    vertices = vec(vertices)
#
#    @tullio p[i, j, e] =
#        @inbounds(begin
#                      c1, c2, c3, c4 = connectivity[e]
#                      ri, rj = $(r[1])[i], $(r[2])[j]
#
#                      #((1 - ri) * (1 - rj) * vertices[c1] +
#                      # (1 + ri) * (1 - rj) * vertices[c2] +
#                      # (1 - ri) * (1 + rj) * vertices[c3] +
#                      # (1 + ri) * (1 + rj) * vertices[c4]) / 4
#
#                      # this is the same as above but
#                      # explicit muladds help with generating symmetric meshes
#                      m1 = muladd(-rj, vertices[c1], vertices[c1]) +
#                           muladd( rj, vertices[c3], vertices[c3])
#                      m2 = muladd(-rj, vertices[c2], vertices[c2]) +
#                           muladd( rj, vertices[c4], vertices[c4])
#
#                      m1 = muladd(-ri, m1, m1) + muladd(ri, m2, m2)
#
#                      m1 / 4
#                  end) (i in axes(p, 1), j in axes(p, 2), e in axes(p, 3))
#
#    return reshape(p, (length(referencecell), length(connectivity)))
#end
#
#function materializemetrics(referencecell::LobattoQuad, points, unwarpedbrick)
#    T = floattype(referencecell)
#    A = arraytype(referencecell)
#    num_cellindices =  length(referencecell)
#    num_cells = last(size(points))
#    faceconn = connectivity(referencecell)[2]
#    num_facesindices = sum(length.(faceconn))
#
#    metrics = fieldarray(undef, (g=SMatrix{2, 2, T, 4}, J=T), A,
#                         (num_cellindices, num_cells))
#    g, J = components(metrics)
#
#    D₁, D₂ = derivatives(referencecell)
#    x₁, x₂ = components(points)
#
#    if unwarpedbrick
#        h = fieldarray(undef, SMatrix{2, 2, T, 4}, A, (num_cellindices, num_cells))
#        h₁₁, h₂₁, h₁₂, h₂₂ = components(h)
#        x₁ = reshape(x₁, size(referencecell)..., num_cells)
#        x₂ = reshape(x₂, size(referencecell)..., num_cells)
#        @tullio avx=false h₁₁[i, e] = (x₁[end, 1, e] - x₁[1, 1, e]) / 2
#        h₂₁ .= 0
#        h₁₂ .= 0
#        @tullio avx=false h₂₂[i, e] = (x₂[1, end, e] - x₂[1, 1, e]) / 2
#    else
#        h = fieldarray(SMatrix{2, 2, T, 4}, (D₁ * x₁, D₁ * x₂, D₂ * x₁, D₂ * x₂))
#    end
#
#    @. J = det(h)
#    @. g = inv(h)
#
#    facemetrics = fieldarray(undef, (n=SVector{2, T}, J=T), A,
#                             (num_facesindices, num_cells))
#    n, fJ = components(facemetrics)
#
#    J = reshape(J, size(referencecell)..., :)
#    g₁₁, g₂₁, g₁₂, g₂₂ = reshape.(components(g), size(referencecell)..., :)
#    n₁, n₂ = components(n)
#    n₁₁, n₁₂, n₁₃, n₁₄ = faceviews(referencecell, n₁)
#    n₂₁, n₂₂, n₂₃, n₂₄ = faceviews(referencecell, n₂)
#
#    @tullio avx=false n₁₁[j, e] = -J[1, j, e] * g₁₁[1, j, e]
#    @tullio avx=false n₂₁[j, e] = -J[1, j, e] * g₁₂[1, j, e]
#
#    @tullio avx=false n₁₂[j, e] = J[end, j, e] * g₁₁[end, j, e]
#    @tullio avx=false n₂₂[j, e] = J[end, j, e] * g₁₂[end, j, e]
#
#    @tullio avx=false n₁₃[i, e] = -J[i, 1, e] * g₂₁[i, 1, e]
#    @tullio avx=false n₂₃[i, e] = -J[i, 1, e] * g₂₂[i, 1, e]
#
#    @tullio avx=false n₁₄[i, e] = J[i, end, e] * g₂₁[i, end, e]
#    @tullio avx=false n₂₄[i, e] = J[i, end, e] * g₂₂[i, end, e]
#
#    # Here we are working around the fact that broadcast things that
#    # n and fJ might alias because they are in the same base array.
#    normn = hypot.(n₁, n₂)
#    fJ .= normn
#    n ./= normn
#
#    return (metrics, facemetrics)
#end
#
#function materializepoints(referencecell::LobattoHex, vertices, connectivity)
#    T = floattype(referencecell)
#    A = arraytype(referencecell)
#    r = vec.(points_1d(referencecell))
#    p = fieldarray(undef, SVector{3, T}, A,
#                   (size(referencecell)..., length(connectivity)))
#
#    connectivity = vec(connectivity)
#    vertices = vec(vertices)
#
#    @tullio p[i, j, k, e] =
#        @inbounds(begin
#                      c1, c2, c3, c4, c5, c6, c7, c8 = connectivity[e]
#                      ri, rj, rk = $(r[1])[i], $(r[2])[j], $(r[3])[k]
#
#                      #((1 - ri) * (1 - rj) * (1 - rk) * vertices[c1] +
#                      # (1 + ri) * (1 - rj) * (1 - rk) * vertices[c2] +
#                      # (1 - ri) * (1 + rj) * (1 - rk) * vertices[c3] +
#                      # (1 + ri) * (1 + rj) * (1 - rk) * vertices[c4] +
#                      # (1 - ri) * (1 - rj) * (1 + rk) * vertices[c5] +
#                      # (1 + ri) * (1 - rj) * (1 + rk) * vertices[c6] +
#                      # (1 - ri) * (1 + rj) * (1 + rk) * vertices[c7] +
#                      # (1 + ri) * (1 + rj) * (1 + rk) * vertices[c8]) / 8
#
#                      # this is the same as above but
#                      # explicit muladds help with generating symmetric meshes
#                      m1 = muladd(-rk, vertices[c1], vertices[c1]) +
#                           muladd( rk, vertices[c5], vertices[c5])
#                      m2 = muladd(-rk, vertices[c3], vertices[c3]) +
#                           muladd( rk, vertices[c7], vertices[c7])
#                      m3 = muladd(-rk, vertices[c2], vertices[c2]) +
#                           muladd( rk, vertices[c6], vertices[c6])
#                      m4 = muladd(-rk, vertices[c4], vertices[c4]) +
#                           muladd( rk, vertices[c8], vertices[c8])
#
#                      m1 = muladd(-rj, m1, m1) + muladd(rj, m2, m2)
#                      m2 = muladd(-rj, m3, m3) + muladd(rj, m4, m4)
#
#                      m1 = muladd(-ri, m1, m1) + muladd(ri, m2, m2)
#
#                      m1 / 8
#                  end) (i in axes(p, 1), j in axes(p, 2), k in axes(p, 3),
#                        e in axes(p, 4))
#
#    return reshape(p, (length(referencecell), length(connectivity)))
#end
#
#function materializemetrics(referencecell::LobattoHex, points, unwarpedbrick)
#    T = floattype(referencecell)
#    A = arraytype(referencecell)
#    num_cellindices =  length(referencecell)
#    num_cells = last(size(points))
#    faceconn = connectivity(referencecell)[2]
#    num_facesindices = sum(length.(faceconn))
#
#    metrics = fieldarray(undef, (g=SMatrix{3, 3, T, 9}, J=T), A,
#                         (num_cellindices, num_cells))
#    g, J = components(metrics)
#
#    D₁, D₂, D₃ = derivatives(referencecell)
#    x₁, x₂, x₃ = components(points)
#
#    if unwarpedbrick
#        h = fieldarray(undef, SMatrix{3, 3, T, 9}, A, (num_cellindices, num_cells))
#        h₁₁, h₂₁, h₃₁, h₁₂, h₂₂, h₃₂, h₁₃, h₂₃, h₃₃ = components(h)
#        x₁ = reshape(x₁, size(referencecell)..., num_cells)
#        x₂ = reshape(x₂, size(referencecell)..., num_cells)
#        x₃ = reshape(x₃, size(referencecell)..., num_cells)
#
#        @tullio avx=false h₁₁[i, e] = (x₁[end, 1, 1, e] - x₁[1, 1, 1, e]) / 2
#        h₂₁ .= 0
#        h₃₁ .= 0
#
#        h₁₂ .= 0
#        @tullio avx=false h₂₂[i, e] = (x₂[1, end, 1, e] - x₂[1, 1, 1, e]) / 2
#        h₃₂ .= 0
#
#        h₁₃ .= 0
#        h₂₃ .= 0
#        @tullio avx=false h₃₃[i, e] = (x₃[1, 1, end, e] - x₃[1, 1, 1, e]) / 2
#    else
#        h = fieldarray(SMatrix{3, 3, T, 9}, (D₁ * x₁, D₁ * x₂, D₁ * x₃,
#                                             D₂ * x₁, D₂ * x₂, D₂ * x₃,
#                                             D₃ * x₁, D₃ * x₂, D₃ * x₃))
#    end
#
#    @. J = det(h)
#
#    if unwarpedbrick
#        @. g = inv(h)
#    else
#        # Instead of
#        # ```julia
#        # @. g = inv(h)
#        # ```
#        # we are using the curl invariant formulation of Kopriva, equation (37) of
#        # <https://doi.org/10.1007/s10915-005-9070-8>.
#
#        h₁₁, h₂₁, h₃₁, h₁₂, h₂₂, h₃₂, h₁₃, h₂₃, h₃₃ = components(h)
#
#        xh₁₁ = x₂ .* h₃₁ .- x₃ .* h₂₁
#        xh₂₁ = x₃ .* h₁₁ .- x₁ .* h₃₁
#        xh₃₁ = x₁ .* h₂₁ .- x₂ .* h₁₁
#
#        xh₁₂ = x₂ .* h₃₂ .- x₃ .* h₂₂
#        xh₂₂ = x₃ .* h₁₂ .- x₁ .* h₃₂
#        xh₃₂ = x₁ .* h₂₂ .- x₂ .* h₁₂
#
#        xh₁₃ = x₂ .* h₃₃ .- x₃ .* h₂₃
#        xh₂₃ = x₃ .* h₁₃ .- x₁ .* h₃₃
#        xh₃₃ = x₁ .* h₂₃ .- x₂ .* h₁₃
#
#        g₁₁, g₂₁, g₃₁, g₁₂, g₂₂, g₃₂, g₁₃, g₂₃, g₃₃ = components(g)
#
#        g₁₁ .= (D₂ * xh₁₃ .- D₃ * xh₁₂) ./ (2 .* J)
#        g₂₁ .= (D₃ * xh₁₁ .- D₁ * xh₁₃) ./ (2 .* J)
#        g₃₁ .= (D₁ * xh₁₂ .- D₂ * xh₁₁) ./ (2 .* J)
#
#        g₁₂ .= (D₂ * xh₂₃ .- D₃ * xh₂₂) ./ (2 .* J)
#        g₂₂ .= (D₃ * xh₂₁ .- D₁ * xh₂₃) ./ (2 .* J)
#        g₃₂ .= (D₁ * xh₂₂ .- D₂ * xh₂₁) ./ (2 .* J)
#
#        g₁₃ .= (D₂ * xh₃₃ .- D₃ * xh₃₂) ./ (2 .* J)
#        g₂₃ .= (D₃ * xh₃₁ .- D₁ * xh₃₃) ./ (2 .* J)
#        g₃₃ .= (D₁ * xh₃₂ .- D₂ * xh₃₁) ./ (2 .* J)
#    end
#
#    facemetrics = fieldarray(undef, (n=SVector{3, T}, J=T), A,
#                             (num_facesindices, num_cells))
#    n, fJ = components(facemetrics)
#
#    J = reshape(J, size(referencecell)..., :)
#    g₁₁, g₂₁, g₃₁, g₁₂, g₂₂, g₃₂, g₁₃, g₂₃, g₃₃ =
#        reshape.(components(g), size(referencecell)..., :)
#
#    n₁, n₂, n₃ = components(n)
#    n₁₁, n₁₂, n₁₃, n₁₄, n₁₅, n₁₆ = faceviews(referencecell, n₁)
#    n₂₁, n₂₂, n₂₃, n₂₄, n₂₅, n₂₆ = faceviews(referencecell, n₂)
#    n₃₁, n₃₂, n₃₃, n₃₄, n₃₅, n₃₆ = faceviews(referencecell, n₃)
#
#    @tullio avx=false n₁₁[j, k, e] = -J[  1,   j,   k, e] * g₁₁[  1,   j,   k, e]
#    @tullio avx=false n₁₂[j, k, e] =  J[end,   j,   k, e] * g₁₁[end,   j,   k, e]
#    @tullio avx=false n₁₃[i, k, e] = -J[  i,   1,   k, e] * g₂₁[  i,   1,   k, e]
#    @tullio avx=false n₁₄[i, k, e] =  J[  i, end,   k, e] * g₂₁[  i, end,   k, e]
#    @tullio avx=false n₁₅[i, j, e] = -J[  i,   j,   1, e] * g₃₁[  i,   j,   1, e]
#    @tullio avx=false n₁₆[i, j, e] =  J[  i,   j, end, e] * g₃₁[  i,   j, end, e]
#
#    @tullio avx=false n₂₁[j, k, e] = -J[  1,   j,   k, e] * g₁₂[  1,   j,   k, e]
#    @tullio avx=false n₂₂[j, k, e] =  J[end,   j,   k, e] * g₁₂[end,   j,   k, e]
#    @tullio avx=false n₂₃[i, k, e] = -J[  i,   1,   k, e] * g₂₂[  i,   1,   k, e]
#    @tullio avx=false n₂₄[i, k, e] =  J[  i, end,   k, e] * g₂₂[  i, end,   k, e]
#    @tullio avx=false n₂₅[i, j, e] = -J[  i,   j,   1, e] * g₃₂[  i,   j,   1, e]
#    @tullio avx=false n₂₆[i, j, e] =  J[  i,   j, end, e] * g₃₂[  i,   j, end, e]
#
#    @tullio avx=false n₃₁[j, k, e] = -J[  1,   j,   k, e] * g₁₃[  1,   j,   k, e]
#    @tullio avx=false n₃₂[j, k, e] =  J[end,   j,   k, e] * g₁₃[end,   j,   k, e]
#    @tullio avx=false n₃₃[i, k, e] = -J[  i,   1,   k, e] * g₂₃[  i,   1,   k, e]
#    @tullio avx=false n₃₄[i, k, e] =  J[  i, end,   k, e] * g₂₃[  i, end,   k, e]
#    @tullio avx=false n₃₅[i, j, e] = -J[  i,   j,   1, e] * g₃₃[  i,   j,   1, e]
#    @tullio avx=false n₃₆[i, j, e] =  J[  i,   j, end, e] * g₃₃[  i,   j, end, e]
#
#    # Here we are working around the fact that broadcast thinks that
#    # n and fJ might alias because they are in the same base array.
#    normn = norm.(n)
#    fJ .= normn
#    n ./= normn
#
#    return (metrics, facemetrics)
#end
