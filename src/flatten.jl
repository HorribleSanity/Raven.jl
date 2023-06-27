# The following code was taken and modified from the following packages:
#  - <https://github.com/JuliaObjects/ConstructionBase.jl>
#  - <https://github.com/rafaqz/Flatten.jl>
#
# ConstructionBase.jl is distributed under the following license.
#
# > Copyright (c) 2019 Takafumi Arakaki, Rafael Schouten, Jan Weidner
# >
# > Permission is hereby granted, free of charge, to any person obtaining
# > a copy of this software and associated documentation files (the
# > "Software"), to deal in the Software without restriction, including
# > without limitation the rights to use, copy, modify, merge, publish,
# > distribute, sublicense, > and/or sell copies of the Software, and to
# > permit persons to whom the Software is furnished to do so, subject to
# > the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# > SOFTWARE OR THE USE OR OTHER
#
# Flatten.jl is distributed under the following license.
#
# > Copyright (c) 2018: Rafael Schouten and Robin Deits.
# >
# > Permission is hereby granted, free of charge, to any person obtaining
# > a copy of this software and associated documentation files (the
# > "Software"), to deal in the Software without restriction, including
# > without limitation the rights to use, copy, modify, merge, publish,
# > distribute, sublicense, and/or sell copies of the Software, and to
# > permit persons to whom the Software is furnished to do so, subject to
# > the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

const USE = Real

_fieldnames(::Type{<:Type}) = ()
_fieldnames(::Type{T}) where {T} = fieldnames(T)

# Generalized nested structure walker
function nested(name, ::Type{T}, ::Type{U}, expr_builder, expr_combiner) where {T,U}
    expr_combiner(T, [Expr(:..., expr_builder(name, T, U, fn)) for fn in _fieldnames(T)])
end

function flatten_builder(name, ::Type{T}, ::Type{U}, fname) where {T,U}
    newname = :(getfield($name, $(QuoteNode(fname))))
    if fieldtype(T, fname) <: U
        return Expr(:tuple, newname)
    else
        return nested(newname, fieldtype(T, fname), U, flatten_builder, flatten_combiner)
    end
end

function flatten_combiner(_, expressions)
    Expr(:tuple, expressions...)
end

function flatten_expr(::Type{T}, ::Type{U}) where {T,U}
    nested(:(obj), T, U, flatten_builder, flatten_combiner)
end

"""
    flatten(obj, use=Real)

Flattens a hierarchical type to a tuple with elements of type `use`.

# Examples

```jldoctest
julia> flatten((a=(Complex(1, 2), 3), b=4))
(1, 2, 3, 4)

```

To convert the tuple to a vector, simply use [flatten(x)...], or
using static arrays to avoid allocations: `SVector(flatten(x))`.
"""
@inline @generated function flatten(obj::T, use::Type{U} = USE) where {T,U}
    if T <: U
        return :((obj,))
    else
        return flatten_expr(T, U)
    end
end

"""
    constructorof(T::Type) -> constructor

Return an object `constructor` that can be used to construct objects of
type `T` from their field values. Typically, `constructor` will be the
type `T` with all parameters removed:

```jldoctest
julia> struct T{A,B}
           a::A
           b::B
       end

julia> Raven.constructorof(T{Int,Int})
T

```

The returned constructor is used to `unflatten` objects hierarchical
types from a list of their values. For example, in this case `T(1,2)`
constructs an object where `T.a==1` an `T.b==2`.

The method `constructorof` should be defined for types that are not
constructed from a tuple of their values.
"""
function constructorof end

constructorof(::Type{<:Tuple}) = tuple
constructorof(::Type{<:Complex}) = complex
constructorof(::Type{<:NamedTuple{names}}) where {names} = NamedTupleConstructor{names}()
struct NamedTupleConstructor{names} end
@inline function (::NamedTupleConstructor{names})(args...) where {names}
    NamedTuple{names}(args)
end
constructorof(::Type{T}) where {T<:StaticArray} = T
@generated function constructorof(::Type{T}) where {T}
    getfield(parentmodule(T), nameof(T))
end

"""
    unflatten(T::Type, data, use::Type=Real)

Construct an object from Tuple or Vector `data` and a Type `T`. The `data`
should be at least as long as the queried fields (of type `use`) in `T`.

# Examples

```julia
julia> unflatten(Tuple{Tuple{Int,Int},Complex{Int,Int}}, (1, 2, 3, 4))
((1, 2), 3 + 4im)

```
"""
function unflatten end
@inline Base.@propagate_inbounds unflatten(datatype, data, use::Type = USE) =
    _unflatten(datatype, data, use, 1)[1]

# Internal type used to generate constructor expressions.
# Represents a bi-directional (doubly linked) type tree where
# child nodes correspond to fields of composite types.
mutable struct TypeNode{T,TChildren}
    type::Type{T}
    name::Union{Int,Symbol}
    parent::Union{Missing,TypeNode}
    children::TChildren
    TypeNode(
        type::Type{T},
        name::Union{Int,Symbol},
        children::Union{Missing,<:Tuple{Vararg{TypeNode}}} = missing,
    ) where {T} = new{T,typeof(children)}(type, name, missing, children)
end
function _buildtree(::Type{T}, name) where {T}
    if isabstracttype(T) || isa(T, Union) || isa(T, UnionAll)
        return TypeNode(T, name)
    elseif T <: Type # treat meta types as leaf nodes
        return TypeNode(T, name, ())
    else
        names = fieldnames(T)
        types = fieldtypes(T)
        children = map(_buildtree, types, names)
        node = TypeNode(T, name, children)
        # set parent field on children
        for child in children
            child.parent = node
        end
        return node
    end
end
# Recursive accessor (getfield) expression builder
_accessor_expr(::Missing, child::TypeNode) = :obj
_accessor_expr(parent::TypeNode, child::TypeNode) =
    Expr(:call, :getfield, _accessor_expr(parent.parent, parent), QuoteNode(child.name))
# Recursive construct expression builder;
# Case 1: Leaf node (i.e., no fields)
_unflatten_expr(node::TypeNode{T,Tuple{}}, use::Type{U}) where {T,U} =
    :(($(_accessor_expr(node.parent, node)), n))
# Case 2: Matched type; value is taken from `data` at index `n`;
_unflatten_expr(node::TypeNode{<:U}, use::Type{U}) where {U} = __unflatten_expr(node, U)
_unflatten_expr(node::TypeNode{<:U,Tuple{}}, use::Type{U}) where {U} =
    __unflatten_expr(node, U)
function __unflatten_expr(node::TypeNode{<:U}, use::Type{U}) where {U}
    :(data[n], n + 1)
end
# Case 3: Composite type fields (i.e., with fields) are constructed recursively;
# Recursion is used only at compile time in building the constructor expression.
# This ensures type stability.
function _unflatten_expr(node::TypeNode{T}, ::Type{U}) where {T,U}
    expr = Expr(:block)
    argnames = []
    # Generate constructor expression for each child/field
    for child in node.children
        flattened_expr = _unflatten_expr(child, U)
        accessor_expr = _accessor_expr(node, child)
        name = gensym("arg")
        child_expr = quote
            ($name, n) = $flattened_expr
        end
        push!(expr.args, child_expr)
        push!(argnames, name)
    end
    # Combine into constructor call
    callexpr = Expr(:call, :(constructorof($T)), argnames...)
    # Return result along with current `data` index, `n`
    push!(expr.args, :(($callexpr, n)))
    return expr
end
@inline Base.@propagate_inbounds @generated function _unflatten(
    ::Type{T},
    data,
    use::Type{U},
    n,
) where {T,U}
    _unflatten_expr(_buildtree(T, :obj), U)
end
