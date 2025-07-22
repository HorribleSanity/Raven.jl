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
constants(::AbstractBalanceLaw{FT,D,S,C}) where {FT,D,S,C} = C

auxiliary(law::AbstractBalanceLaw, x⃗) = SVector(nothing)

function flux end
function wavespeed end
boundarystate(::AbstractBalanceLaw, ::AbstractProblem, n⃗, q⁻, aux⁻, tag) = q⁻, aux⁻
source!(::AbstractBalanceLaw, dq, q, aux, dim, directions) = nothing
source!(::AbstractBalanceLaw, ::AbstractProblem, dq, q, aux, dim, directions) = nothing
nonconservative_term!(::AbstractBalanceLaw, dq, q, aux, directions, dim) = nothing
function Bennu.fieldarray(init, law::AbstractBalanceLaw, grid::Bennu.AbstractGrid)
    FT = eltype(law)
    nstate = numberofstates(law)
    return fieldarray(init, SVector{nstate,FT}, grid)
end

function entropy end
function entropyvariables end
