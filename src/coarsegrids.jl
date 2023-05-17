abstract type AbstractCoarseGrid end

struct BrickGrid{T,C,D,M} <: AbstractCoarseGrid
    connectivity::C
    coordinates::D
    mapping::M
end

connectivity(g::BrickGrid) = g.connectivity
coordinates(g::BrickGrid) = g.coordinates
mapping(g::BrickGrid) = g.mapping
function vertices(g::BrickGrid{T}) where {T}
    conn = connectivity(g)
    coords = coordinates(g)
    m = mapping(g)
    indices =
        GC.@preserve conn convert.(Tuple{Int,Int,Int}, P4estTypes.unsafe_vertices(conn))
    if conn isa P4estTypes.Connectivity{4}
        verts = [SVector(coords[1][i[1]+1], coords[2][i[2]+1]) for i in indices]
    else
        verts = [
            SVector(coords[1][i[1]+1], coords[2][i[2]+1], coords[3][i[3]+1]) for
            i in indices
        ]
    end
    return m.(verts)
end
function cells(g::BrickGrid)
    conn = connectivity(g)
    GC.@preserve conn map.(x -> x + 1, P4estTypes.unsafe_trees(conn))
end

function brick(
    T::Type,
    n::Tuple{Integer,Integer},
    p::Tuple{Bool,Bool} = (false, false);
    coordinates = (zero(T):n[1], zero(T):n[2]),
    mapping = identity,
)
    if length.(coordinates) != n .+ 1
        throw(
            DimensionMismatch(
                "coordinates lengths $(length.(coordinates)) should correspond to the number of trees + 1, $(n .+ 1)",
            ),
        )
    end


    connectivity = P4estTypes.brick(n, p)

    return BrickGrid{T,typeof(connectivity),typeof(coordinates),typeof(mapping)}(
        connectivity,
        coordinates,
        mapping,
    )
end

function brick(
    T::Type,
    n::Tuple{Integer,Integer,Integer},
    p::Tuple{Bool,Bool,Bool} = (false, false, false);
    coordinates = (zero(T):n[1], zero(T):n[2], zero(T):n[3]),
    mapping = identity,
)
    if length.(coordinates) != n .+ 1
        throw(
            DimensionMismatch(
                "Coordinate lengths $(length.(coordinates)) should correspond to the number of trees + 1, $(n .+ 1)",
            ),
        )
    end

    connectivity = P4estTypes.brick(n, p)

    return BrickGrid{T,typeof(connectivity),typeof(coordinates),typeof(mapping)}(
        connectivity,
        coordinates,
        mapping,
    )
end

function brick(n::Tuple{Integer,Integer}, p::Tuple{Bool,Bool} = (false, false); kwargs...)
    return brick(Float64, n, p, kwargs...)
end

function brick(
    n::Tuple{Integer,Integer,Integer},
    p::Tuple{Bool,Bool,Bool} = (false, false, false);
    kwargs...,
)
    return brick(Float64, n, p, kwargs...)
end

brick(l::Integer, m::Integer, p::Bool = false, q::Bool = false; kwargs...) =
    brick(Float64, (l, m), (p, q); kwargs...)
brick(T::Type, l::Integer, m::Integer, p::Bool = false, q::Bool = false; kwargs...) =
    brick(T, (l, m), (p, q); kwargs...)

function brick(
    l::Integer,
    m::Integer,
    n::Integer,
    p::Bool = false,
    q::Bool = false,
    r::Bool = false;
    kwargs...,
)
    return brick(Float64, (l, m, n), (p, q, r); kwargs...)
end

function brick(
    T::Type,
    l::Integer,
    m::Integer,
    n::Integer,
    p::Bool = false,
    q::Bool = false,
    r::Bool = false;
    kwargs...,
)
    return brick(T, (l, m, n), (p, q, r); kwargs...)
end


@recipe function f(coarsegrid::BrickGrid)
    cs = cells(coarsegrid)
    vs = vertices(coarsegrid)

    xlims = extrema(getindex.(vs, 1))
    ylims = extrema(getindex.(vs, 2))
    zlims = try
        extrema(getindex.(vs, 3))
    catch
        (zero(eltype(xlims)), zero(eltype(xlims)))
    end
    isconstz = zlims[1] == zlims[2]

    xlabel --> "x"
    ylabel --> "y"
    zlabel --> "z"

    aspect_ratio --> :equal
    legend --> false
    grid --> false

    @series begin
        seriestype --> :path
        linecolor --> :gray
        linewidth --> 1

        x = []
        y = []
        z = []
        if length(first(cs)) == 4
            for c in cs
                for i in (1, 2, 4, 3, 1)

                    xs = vs[c[i]]

                    push!(x, xs[1])
                    push!(y, xs[2])
                    if !isconstz
                        push!(z, xs[3])
                    end
                end

                push!(x, NaN)
                push!(y, NaN)
                if !isconstz
                    push!(z, NaN)
                end
            end
        elseif length(first(cs)) == 8
            for c in cs
                for j in (0, 4)
                    for i in (1 + j, 2 + j, 4 + j, 3 + j, 1 + j)
                        xi, yi, zi = vs[c[i]]

                        push!(x, xi)
                        push!(y, yi)
                        push!(z, zi)
                    end

                    push!(x, NaN)
                    push!(y, NaN)
                    push!(z, NaN)
                end

                for j = 0:3
                    for i in (1 + j, 5 + j)
                        xi, yi, zi = vs[c[i]]

                        push!(x, xi)
                        push!(y, yi)
                        push!(z, zi)
                    end

                    push!(x, NaN)
                    push!(y, NaN)
                    push!(z, NaN)
                end
            end
        end
        if isconstz
            x, y
        else
            x, y, z
        end
    end
end
