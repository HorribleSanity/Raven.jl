abstract type AbstractCell{S<:Tuple,T,A<:AbstractArray,N} end

floattype(::Type{<:AbstractCell{S,T}}) where {S,T} = T
arraytype(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} = A
Base.ndims(::Type{<:AbstractCell{S,T,A,N}}) where {S,T,A,N} = N
Base.size(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} = size_to_tuple(S)
Base.size(::Type{<:AbstractCell{S,T,A}}, i::Integer) where {S,T,A} = size_to_tuple(S)[i]
Base.length(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} = tuple_prod(S)
Base.strides(::Type{<:AbstractCell{S,T,A}}) where {S,T,A} =
    Base.size_to_strides(1, size_to_tuple(S)...)

floattype(cell::AbstractCell) = floattype(typeof(cell))
arraytype(cell::AbstractCell) = arraytype(typeof(cell))
Base.ndims(cell::AbstractCell) = Base.ndims(typeof(cell))
Base.size(cell::AbstractCell) = Base.size(typeof(cell))
Base.size(cell::AbstractCell, i::Integer) = Base.size(typeof(cell), i)
Base.length(cell::AbstractCell) = Base.length(typeof(cell))
Base.strides(cell::AbstractCell) = Base.strides(typeof(cell))
