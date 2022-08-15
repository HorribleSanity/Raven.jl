abstract type AbstractCell{T,A<:AbstractArray,S<:Tuple,N} end

floattype(::Type{<:AbstractCell{T}}) where {T} = T
arraytype(::Type{<:AbstractCell{T,A}}) where {T,A} = A
Base.ndims(::Type{<:AbstractCell{T,A,S,N}}) where {T,A,S,N} = N
Base.size(::Type{<:AbstractCell{T,A,S}}) where {T,A,S} = size_to_tuple(S)
Base.length(::Type{<:AbstractCell{T,A,S}}) where {T,A,S} = tuple_prod(S)
Base.strides(::Type{<:AbstractCell{T,A,S}}) where {T,A,S} =
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

    setprecision(oldprecision)

    return map(
        Array{T},
        (
            points = points,
            weights = weights,
            derivative = derivative,
            toequallyspaced = toequallyspaced,
        ),
    )
end

struct LobattoCell{T,A,S,N,O,P,D,M,FM,E,C} <: AbstractCell{T,A,S,N}
    points_1d::O
    weights_1d::O
    points::P
    derivatives::D
    mass::M
    facemass::FM
    toequallyspaced::E
    connectivity::C
end

function Base.show(io::IO, ::LobattoCell{T,A,N}) where {T,A,N}
    print(io, "LobattoCell{")
    Base.show(io, T)
    print(io, ", ")
    Base.show(io, A)
    print(io, ", ")
    Base.show(io, N)
    print(io, "}")
end

function LobattoCell{T,A}(dims...) where {T,A}
    N = length(dims)
    if all(dims[1] .== dims)
        oall = adapt(A, lobattooperators_1d(T, first(dims)))
        o = ntuple(i -> oall, N)
    else
        o = ntuple(i -> adapt(A, lobattooperators_1d(T, dims[i])), N)
    end

    points_1d = ntuple(N) do i
        return reshape(o[i].points, ntuple(j -> ifelse(i == j, dims[i], 1), N))
    end
    weights_1d = ntuple(N) do i
        return reshape(o[i].weights, ntuple(j -> ifelse(i == j, dims[i], 1), N))
    end

    points = vec(SVector.(points_1d...))
    # TODO Should we use a struct of arrays style layout?
    # if isbitstype(T) && N > 1
    #     # Setup struct of arrays style layout of the points
    #     points = reinterpret(reshape, T, points)
    #     points = permutedims(points, (2, 1))
    #     points = PermutedDimsArray(points, (2, 1))
    #     points = reinterpret(reshape, SVector{N, T}, points)
    # end

    derivatives = ntuple(N) do i
        tup = ntuple(j -> ifelse(i == j, o[i].derivative, Eye{T}(dims[j])), N)
        return Kron(reverse(tup))
    end

    mass = Diagonal(vec(.*(weights_1d...)))

    facemass = if N == 1
        adapt(A, Diagonal([T(1), T(1)]))
    elseif N == 2
        ω1, ω2 = weights_1d
        Diagonal(vcat(repeat(vec(ω2), 2), repeat(vec(ω1), 2)))
    elseif N == 3
        ω1, ω2, ω3 = weights_1d
        Diagonal(
            vcat(
                repeat(vec(ω2 .* ω3), 2),
                repeat(vec(ω1 .* ω3), 2),
                repeat(vec(ω1 .* ω2), 2),
            ),
        )
    end

    toequallyspaced = Kron(reverse(ntuple(i -> o[i].toequallyspaced, N)))

    connectivity = adapt(A, materializeconnectivity(LobattoCell, dims...))

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        connectivity,
    )
    LobattoCell{T,A,Tuple{dims...},N,typeof.(args[2:end])...}(args...)
end

LobattoCell{T}(dims...) where {T} = LobattoCell{T,Array}(dims...)
LobattoCell(dims...) = LobattoCell{Float64}(dims...)

function Base.similar(::LobattoCell{T,A}, dims...) where {T,A}
    LobattoCell{T,A}(dims...)
end

function Adapt.adapt_structure(to, cell::LobattoCell{T,A,S,N}) where {T,A,S,N}
    names = fieldnames(LobattoCell)
    args = ntuple(j -> adapt(to, getfield(cell, names[j])), length(names))
    B = arraytype(to)

    LobattoCell{T,B,S,N,typeof.(args[2:end])...}(args...)
end

const LobattoLine{T,A} = LobattoCell{T,A,Tuple{B}} where {B}
const LobattoQuad{T,A} = LobattoCell{T,A,Tuple{B,C}} where {B,C}
const LobattoHex{T,A} = LobattoCell{T,A,Tuple{B,C,D}} where {B,C,D}

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
connectivity(cell::LobattoCell) = cell.connectivity
degrees(cell::LobattoCell) = size(cell) .- 1

celltype_vtk(::LobattoLine) = VTKCellTypes.VTK_LAGRANGE_CURVE
celltype_vtk(::LobattoQuad) = VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL
celltype_vtk(::LobattoHex) = VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON

function connectivity_vtk(cell::LobattoLine)
    L = LinearIndices(size(cell))
    return [
        L[begin],      # corners
        L[end],
        L[begin+1:end-1]..., # interior
    ]
end

function connectivity_vtk(cell::LobattoQuad)
    L = LinearIndices(size(cell))
    return [
        L[begin, begin], # corners
        L[end, begin],
        L[end, end],
        L[begin, end],
        L[begin+1:end-1, begin]..., # edges
        L[end, begin+1:end-1]...,
        L[begin+1:end-1, end]...,
        L[begin, begin+1:end-1]...,
        L[begin+1:end-1, begin+1:end-1]..., # interior
    ]
end

function connectivity_vtk(cell::LobattoHex)
    L = LinearIndices(size(cell))
    return [
        L[begin, begin, begin], # corners
        L[end, begin, begin],
        L[end, end, begin],
        L[begin, end, begin],
        L[begin, begin, end],
        L[end, begin, end],
        L[end, end, end],
        L[begin, end, end],
        L[begin+1:end-1, begin, begin]..., # edges
        L[end, begin+1:end-1, begin]...,
        L[begin+1:end-1, end, begin]...,
        L[begin, begin+1:end-1, begin]...,
        L[begin+1:end-1, begin, end]...,
        L[end, begin+1:end-1, end]...,
        L[begin+1:end-1, end, end]...,
        L[begin, begin+1:end-1, end]...,
        L[begin, begin, begin+1:end-1]...,
        L[end, begin, begin+1:end-1]...,
        L[begin, end, begin+1:end-1]...,
        L[end, end, begin+1:end-1]...,
        L[begin, begin+1:end-1, begin+1:end-1]..., # faces
        L[end, begin+1:end-1, begin+1:end-1]...,
        L[begin+1:end-1, begin, begin+1:end-1]...,
        L[begin+1:end-1, end, begin+1:end-1]...,
        L[begin+1:end-1, begin+1:end-1, begin]...,
        L[begin+1:end-1, begin+1:end-1, end]...,
        L[begin+1:end-1, begin+1:end-1, begin+1:end-1]..., # interior
    ]
end

function materializeconnectivity(::Type{<:LobattoCell}, L::Integer)
    indices = collect(LinearIndices((L,)))

    conn = (
        (indices,), # edge
        ( # corners
            indices[begin],
            indices[end],
        ),
    )

    return conn
end

function materializeconnectivity(::Type{<:LobattoCell}, L::Integer, M::Integer)
    indices = collect(LinearIndices((L, M)))

    conn = (
        (indices,), # face
        ( # edges
            (indices[begin, begin:end]),
            (indices[end, begin:end]),
            (indices[begin:end, begin]),
            (indices[begin:end, end]),
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

function materializeconnectivity(::Type{<:LobattoCell}, L::Integer, M::Integer, N::Integer)
    indices = collect(LinearIndices((L, M, N)))

    conn = (
        (indices,), # volume
        ( # faces
            (indices[begin, begin:end, begin:end]),
            (indices[end, begin:end, begin:end]),
            (indices[begin:end, begin, begin:end]),
            (indices[begin:end, end, begin:end]),
            (indices[begin:end, begin:end, begin]),
            (indices[begin:end, begin:end, end]),
        ),
        ( # edges
            (indices[begin, begin, begin:end]),
            (indices[end, begin, begin:end]),
            (indices[begin, end, begin:end]),
            (indices[end, end, begin:end]),
            (indices[begin, begin:end, begin]),
            (indices[end, begin:end, begin]),
            (indices[begin, begin:end, end]),
            (indices[end, begin:end, end]),
            (indices[begin:end, begin, begin]),
            (indices[begin:end, end, begin]),
            (indices[begin:end, begin, end]),
            (indices[begin:end, end, end]),
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
