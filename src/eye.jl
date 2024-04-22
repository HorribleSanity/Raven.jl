struct Eye{T} <: AbstractArray{T,2}
    N::Int
end

Base.size(eye::Eye{T}) where {T} = (eye.N, eye.N)
Base.IndexStyle(::Eye) = IndexCartesian()
@inline function Base.getindex(eye::Eye{T}, I::Vararg{Int,2}) where {T}
    @boundscheck checkbounds(eye, I...)
    return (I[1] == I[2]) ? one(T) : zero(T)
end
