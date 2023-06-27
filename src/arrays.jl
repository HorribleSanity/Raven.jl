# The parenttype function and get_backend were adapted from ArrayInterfaceCore
# with the following license.
# MIT License
# 
# Copyright (c) 2022 SciML
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
    parent_type(::Type{T}) -> Type

Returns the parent array that type `T` wraps.
"""
parenttype(x) = parenttype(typeof(x))
parenttype(::Type{Symmetric{T,S}}) where {T,S} = S
parenttype(@nospecialize T::Type{<:PermutedDimsArray}) = fieldtype(T, :parent)
parenttype(@nospecialize T::Type{<:Adjoint}) = fieldtype(T, :parent)
parenttype(@nospecialize T::Type{<:Transpose}) = fieldtype(T, :parent)
parenttype(@nospecialize T::Type{<:SubArray}) = fieldtype(T, :parent)
parenttype(@nospecialize T::Type{<:Base.ReinterpretArray}) = fieldtype(T, :parent)
parenttype(@nospecialize T::Type{<:Base.ReshapedArray}) = fieldtype(T, :parent)
parenttype(@nospecialize T::Type{<:Union{Base.Slice,Base.IdentityUnitRange}}) =
    fieldtype(T, :indices)
parenttype(::Type{Diagonal{T,V}}) where {T,V} = V
parenttype(T::Type) = T

"""
    get_backend(::Type{T}) -> Type

Returns the KernelAbstractions backend to use with kernels where `A` is an
argument.
"""
get_backend(A) = get_backend(typeof(A))
get_backend(::Type) = nothing
get_backend(::Type{T}) where {T<:Array} = CPU(; static = true)
get_backend(::Type{T}) where {T<:AbstractArray} = get_backend(parenttype(T))

arraytype(A) = arraytype(typeof(A))
arraytype(::Type) = nothing
arraytype(::Type{T}) where {T<:Array} = Array
arraytype(::Type{T}) where {T<:AbstractArray} = arraytype(parenttype(T))

"""
    pin(T::Type, A::Array)

    Pins the host array A for copying to arrays of type T
"""
pin(::Type, A::Array) = A

"""
    numbercontiguous(T, A; by = identity)

Renumbers `A` contiguously in an `Array{T}` and returns it. The function
`by` is a mapping for the elements of `A` used during element comparison,
similar to `sort`.

# Examples
```jldoctest
julia> Raven.numbercontiguous(Int32, [13, 4, 5, 1, 5])
5-element Vector{Int32}:
 4
 2
 3
 1
 3

julia> Raven.numbercontiguous(Int32, [13, 4, 5, 1, 5]; by = x->-x)
5-element Vector{Int32}:
 1
 3
 2
 4
 2
```
"""
function numbercontiguous(::Type{T}, A; by = identity) where {T}
    p = sortperm(vec(A); by = by)
    notequalprevious = fill!(similar(p, Bool), false)

    for i in Iterators.drop(eachindex(p), 1)
        notequalprevious[i] = by(A[p[i]]) != by(A[p[i-1]])
    end

    B = similar(A, T)
    B[p] .= cumsum(notequalprevious) .+ 1

    return B
end
