export numberofstates, constants
export AbstractProblem
export problem


abstract type AbstractBalanceLaw{FT,D,S,C} end

abstract type AbstractProblem end
struct DummyProblem <: AbstractProblem end
problem(law::AbstractBalanceLaw) = law.problem

Base.eltype(::AbstractBalanceLaw{FT}) where {FT} = FT
Base.ndims(::AbstractBalanceLaw{FT,D}) where {FT,D} = D
numberofstates(::AbstractBalanceLaw{FT,D,S}) where {FT,D,S} = S
typeofstate(::AbstractBalanceLaw{FT,D,S}) where {FT,D,S} = SVector{S,FT}
constants(::AbstractBalanceLaw{FT,D,S,C}) where {FT,D,S,C} = C

auxiliary(::AbstractBalanceLaw{FT}, x⃗) where {FT} = SVector{0,FT}()

function flux end
function wavespeed end
boundarystate(::AbstractBalanceLaw, ::AbstractProblem, n⃗, q⁻, aux⁻, tag) = q⁻, aux⁻
source!(::AbstractBalanceLaw, dq, q, aux, dim, directions, time) = nothing
function source!(::AbstractBalanceLaw, ::AbstractProblem, dq, q, aux, dim, directions, time)
    return
end
nonconservative_term!(::AbstractBalanceLaw, dq, q, aux, directions, dim) = nothing
function Raven.GridArray(init, law::AbstractBalanceLaw, grid::Raven.Grid)
    return GridArray{typeofstate(law)}(init, grid)
end

function entropy end
function entropyvariables end
