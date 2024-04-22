abstract type AbstractCell{T,A<:AbstractArray,N} end

floattype(::Type{<:AbstractCell{T}}) where {T} = T
arraytype(::Type{<:AbstractCell{T,A}}) where {T,A} = A
Base.ndims(::Type{<:AbstractCell{T,A,N}}) where {T,A,N} = N

floattype(cell::AbstractCell) = floattype(typeof(cell))
arraytype(cell::AbstractCell) = arraytype(typeof(cell))
Base.ndims(cell::AbstractCell) = Base.ndims(typeof(cell))

Base.size(cell::AbstractCell, i::Integer) = size(cell)[i]
Base.length(cell::AbstractCell) = prod(size(cell))
Base.strides(cell::AbstractCell) = Base.size_to_strides(1, size(cell)...)
