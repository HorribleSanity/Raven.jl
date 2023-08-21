function gaussoperators_1d(::Type{T}, M) where {T}
    oldprecision = precision(BigFloat)

    # Increase precision of the type used to compute the 1D operators to help
    # ensure any symmetries.  This is not thread safe so an alternative would
    # be to use ArbNumerics.jl which keeps its precision in the type.
    setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(T))) + 2)))

    points, weights = legendregauss(BigFloat, M)
    derivative = spectralderivative(points)
    equallyspacedpoints = range(-one(BigFloat), stop = one(BigFloat), length = M)
    toequallyspaced = spectralinterpolation(points, equallyspacedpoints)
    tolowerhalf = spectralinterpolation(points, (points .- 1) ./ 2)
    toupperhalf = spectralinterpolation(points, (points .+ 1) ./ 2)

    setprecision(oldprecision)

    points = Array{T}(points)
    weights = Array{T}(weights)
    derivative = Array{T}(derivative)
    toequallyspaced = Array{T}(toequallyspaced)
    tohalves = (Array{T}(tolowerhalf), Array{T}(toupperhalf))

    return (; points, weights, derivative, toequallyspaced, tohalves)
end

struct GaussCell{S,T,A,N,O,P,D,M,FM,E,H} <: AbstractCell{S,T,A,N}
    points_1d::O
    weights_1d::O
    points::P
    derivatives::D
    mass::M
    facemass::FM
    toequallyspaced::E
    tohalves_1d::H
end

function Base.show(io::IO, ::GaussCell{S,T,A}) where {S,T,A}
    print(io, "GaussCell{")
    Base.show(io, S)
    print(io, ", ")
    Base.show(io, T)
    print(io, ", ")
    Base.show(io, A)
    print(io, "}")
end

function GaussCell{Tuple{S1},T,A}() where {S1,T,A}
    o = adapt(A, gaussoperators_1d(T, S1))
    points_1d = (o.points,)
    weights_1d = (o.weights,)

    points = vec(SVector.(points_1d...))
    derivatives = (Kron((o.derivative,)),)
    mass = Diagonal(vec(.*(weights_1d...)))
    facemass = adapt(A, Diagonal([T(1), T(1)]))
    toequallyspaced = Kron((o.toequallyspaced,))
    tohalves_1d = ((o.tohalves[1], o.tohalves[2]),)

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
    )
    GaussCell{Tuple{S1},T,A,1,typeof.(args[2:end])...}(args...)
end

function GaussCell{Tuple{S1,S2},T,A}() where {S1,S2,T,A}
    o = adapt(A, (gaussoperators_1d(T, S1), gaussoperators_1d(T, S2)))

    points_1d = (reshape(o[1].points, (S1, 1)), reshape(o[2].points, (1, S2)))
    weights_1d = (reshape(o[1].weights, (S1, 1)), reshape(o[2].weights, (1, S2)))
    points = vec(SVector.(points_1d...))
    derivatives =
        (Kron((Eye{T,S2}(), o[1].derivative)), Kron((o[2].derivative, Eye{T,S1}())))
    mass = Diagonal(vec(.*(weights_1d...)))
    ω1, ω2 = weights_1d
    facemass = Diagonal(vcat(repeat(vec(ω2), 2), repeat(vec(ω1), 2)))
    toequallyspaced = Kron((o[2].toequallyspaced, o[1].toequallyspaced))
    tohalves_1d =
        ((o[1].tohalves[1], o[1].tohalves[2]), (o[2].tohalves[1], o[2].tohalves[2]))

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
    )
    GaussCell{Tuple{S1,S2},T,A,2,typeof.(args[2:end])...}(args...)
end

function GaussCell{Tuple{S1,S2,S3},T,A}() where {S1,S2,S3,T,A}
    o = adapt(
        A,
        (gaussoperators_1d(T, S1), gaussoperators_1d(T, S2), gaussoperators_1d(T, S3)),
    )

    points_1d = (
        reshape(o[1].points, (S1, 1, 1)),
        reshape(o[2].points, (1, S2, 1)),
        reshape(o[3].points, (1, 1, S3)),
    )
    weights_1d = (
        reshape(o[1].weights, (S1, 1, 1)),
        reshape(o[2].weights, (1, S2, 1)),
        reshape(o[3].weights, (1, 1, S3)),
    )
    points = vec(SVector.(points_1d...))
    derivatives = (
        Kron((Eye{T,S3}(), Eye{T,S2}(), o[1].derivative)),
        Kron((Eye{T,S3}(), o[2].derivative, Eye{T,S1}())),
        Kron((o[3].derivative, Eye{T,S2}(), Eye{T,S1}())),
    )
    mass = Diagonal(vec(.*(weights_1d...)))
    ω1, ω2, ω3 = weights_1d
    facemass = Diagonal(
        vcat(repeat(vec(ω2 .* ω3), 2), repeat(vec(ω1 .* ω3), 2), repeat(vec(ω1 .* ω2), 2)),
    )
    toequallyspaced =
        Kron((o[3].toequallyspaced, o[2].toequallyspaced, o[1].toequallyspaced))
    tohalves_1d = (
        (o[1].tohalves[1], o[1].tohalves[2]),
        (o[2].tohalves[1], o[2].tohalves[2]),
        (o[3].tohalves[1], o[3].tohalves[2]),
    )

    args = (
        points_1d,
        weights_1d,
        points,
        derivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
    )
    GaussCell{Tuple{S1,S2,S3},T,A,3,typeof.(args[2:end])...}(args...)
end

GaussCell{S,T}() where {S,T} = GaussCell{S,T,Array}()
GaussCell{S}() where {S} = GaussCell{S,Float64}()

function Adapt.adapt_structure(to, cell::GaussCell{S,T,A,N}) where {S,T,A,N}
    names = fieldnames(GaussCell)
    args = ntuple(j -> adapt(to, getfield(cell, names[j])), length(names))
    B = arraytype(to)

    GaussCell{S,T,B,N,typeof.(args[2:end])...}(args...)
end

const GaussLine{T,A} = GaussCell{Tuple{B},T,A} where {B}
const GaussQuad{T,A} = GaussCell{Tuple{B,C},T,A} where {B,C}
const GaussHex{T,A} = GaussCell{Tuple{B,C,D},T,A} where {B,C,D}

points_1d(cell::GaussCell) = cell.points_1d
weights_1d(cell::GaussCell) = cell.weights_1d
points(cell::GaussCell) = cell.points
derivatives(cell::GaussCell) = cell.derivatives
function derivatives_1d(cell::GaussCell)
    N = ndims(cell)
    ntuple(i -> cell.derivatives[i].args[N-i+1], Val(N))
end
mass(cell::GaussCell) = cell.mass
facemass(cell::GaussCell) = cell.facemass
toequallyspaced(cell::GaussCell) = cell.toequallyspaced
tohalves_1d(cell::GaussCell) = cell.tohalves_1d
degrees(cell::GaussCell) = size(cell) .- 1
