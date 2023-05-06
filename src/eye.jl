struct Eye{T,N} <: AbstractArray{T,2} end

Base.size(::Eye{T,N}) where {T,N} = (N, N)
Base.IndexStyle(::Eye{T,N}) where {T,N} = IndexCartesian()
function Base.getindex(::Eye{T,N}, I::Vararg{Int,2}) where {T,N}
    return (I[1] == I[2]) ? one(T) : zero(T)
end
