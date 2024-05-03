_numkinds(::Val{2}) = 2
_numkinds(::Val{4}) = 8
#_numkinds(::Val{8}) = 48

struct Orientation{N}
    kind::Int8

    function Orientation{N}(kind) where {N}
        if kind < 1 || kind > _numkinds(Val(N))
            throw(BoundsError(1:_numkinds(Val(N)), kind))
        end
        new{N}(kind)
    end
end

kind(o::Orientation) = o.kind

Base.zero(::Type{Orientation{N}}) where {N} = Orientation{N}(1)

@inline function orientindex(o::Orientation, dims::Dims, I...)
    return orientindex(o, dims, CartesianIndex(I))
end

function orientindices(o::Orientation, dims::Dims)
    return orientindex.(Ref(o), Ref(dims), CartesianIndices(dims))
end

function orientindices(o::Orientation{4}, dims::Dims{2})
    indices = orientindex.(Ref(o), Ref(dims), CartesianIndices(dims))

    if fld1(kind(o), 4) != 1
        indices = reshape(indices, reverse(dims))
    end

    return indices
end

@inline function orientindex(
    o::Orientation{2},
    dims::Dims{1},
    I::CartesianIndex{1},
    ::Bool = false,
)
    @boundscheck Base.checkbounds_indices(Bool, (Base.OneTo(dims[1]),), Tuple(I)) ||
                 throw(BoundsError(o, I))

    index = kind(o) == 1 ? I[1] : dims[1] - I[1] + 1

    return CartesianIndex(index)
end

@inline function orientindex(o::Orientation{4}, dims::Dims{2}, I::CartesianIndex{2})
    @boundscheck if !Base.checkbounds_indices(
        Bool,
        (Base.OneTo(dims[1]), Base.OneTo(dims[2])),
        Tuple(I),
    )
        throw(BoundsError(o, I))
    end

    div, rem = divrem(kind(o) - 0x1, 4)

    @inbounds if div != 0
        li = LinearIndices(dims)[I]
        ci = CartesianIndices(reverse(dims))[li]
        I = CartesianIndex(reverse(Tuple(ci)))
    end

    i1 = rem & 0x1 == 0x1 ? dims[1] - I[1] + 1 : I[1]
    i2 = rem & 0x2 == 0x2 ? dims[2] - I[2] + 1 : I[2]

    return CartesianIndex(i1, i2)
end

# function orientindices(o::Orientation{8}, dims::Dims{3}, invert::Bool = false)
#     is1 = StepRange(1, Int8(1), dims[1])
#     is2 = StepRange(1, Int8(1), dims[2])
#     is3 = StepRange(1, Int8(1), dims[3])
#
#     div, rem = divrem(kind(o) - 0x1, 8)
#
#     if rem & 0x1 == 0x1
#         is1 = reverse(is1)
#     end
#
#     if rem & 0x2 == 0x2
#         is2 = reverse(is2)
#     end
#
#     if rem & 0x4 == 0x4
#         is3 = reverse(is3)
#     end
#
#     indices = (is1, is2, is3)
#
#     if div == 0
#         perm = (1, 2, 3)
#     elseif div == 1
#         perm = (1, 3, 2)
#     elseif div == 2
#         perm = (2, 1, 3)
#     elseif div == 3
#         perm = (2, 3, 1)
#     elseif div == 4
#         perm = (3, 1, 2)
#     elseif div == 5
#         perm = (3, 2, 1)
#     else
#         throw(ArgumentError("Orientation{8}($(kind(o))) is invalid"))
#     end
#
#     if invert
#         indices = getindex.(Ref(indices), perm)
#         perm = invperm(perm)
#     end
#
#     return PermutedDimsArray(CartesianIndices(indices), perm)
# end

Base.inv(p::Orientation{2}) = p

function Base.inv(p::Orientation{4})
    invmap = SA{Int8}[1, 2, 3, 4, 5, 7, 6, 8]
    return @inbounds Orientation{4}(invmap[kind(p)])
end

_permcomposition(::Val{2}) = SA{Int8}[1 2; 2 1]
_permcomposition(::Val{4}) = SA{Int8}[
    1 2 3 4 5 6 7 8
    2 1 4 3 7 8 5 6
    3 4 1 2 6 5 8 7
    4 3 2 1 8 7 6 5
    5 6 7 8 1 2 3 4
    6 5 8 7 3 4 1 2
    7 8 5 6 2 1 4 3
    8 7 6 5 4 3 2 1
]
function Base.:(∘)(p::Orientation{N}, q::Orientation{N}) where {N}
    return @inbounds Orientation{N}(_permcomposition(Val(N))[p.kind, q.kind])
end

function orient(::Val{2}, src::AbstractArray{<:Any,1})
    if size(src) != (2,)
        throw(ArgumentError("Argument src=$src needs to be of size (2,)"))
    end

    k = src[1] ≤ src[2] ? 1 : 2

    return Orientation{2}(k)
end

function orient(::Val{4}, src::AbstractArray{<:Any,2})
    if size(src) != (2, 2)
        throw(ArgumentError("Argument src=$src needs to be of size (2,2)"))
    end

    i = LinearIndices(src)[argmin(src)]

    if i == 1
        k = src[2] <= src[3] ? 1 : 5
    elseif i == 2
        k = src[4] <= src[1] ? 6 : 2
    elseif i == 3
        k = src[1] <= src[4] ? 7 : 3
    else # i == 4
        k = src[3] <= src[2] ? 4 : 8
    end

    return Orientation{4}(k)
end
