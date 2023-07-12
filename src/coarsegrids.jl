abstract type AbstractCoarseGrid end

struct CoarseGrid{C,V,L} <: AbstractCoarseGrid
    connectivity::C
    vertices::V
    cells::L
end

connectivity(g::CoarseGrid) = g.connectivity
vertices(g::CoarseGrid) = g.vertices
cells(g::CoarseGrid) = g.cells

function coarsegrid(vertices, cells::AbstractVector{NTuple{X,T}}) where {X,T}
    conn = P4estTypes.Connectivity{X}(vertices, cells)
    return CoarseGrid{typeof(conn),typeof(vertices),typeof(cells)}(conn, vertices, cells)
end

"""
    function cubeshellgrid(R::Real, nedge::Int)

    This function will construct the CoarseGrid of a cube shell of radius R with nedge^2 cells/trees
    along each face. A cube shell is a 2D connectivity.
"""
function cubeshellgrid(R::Real, nedge::Int)
    vert_temp = zeros(SVector{3,Float64}, 8)

    vert_temp[1] = SVector(+R, +R, -R)
    vert_temp[2] = SVector(+R, -R, -R)
    vert_temp[3] = SVector(+R, +R, +R)
    vert_temp[4] = SVector(+R, -R, +R)

    vert_temp[5] = SVector(-R, +R, -R)
    vert_temp[6] = SVector(-R, -R, -R)
    vert_temp[7] = SVector(-R, +R, +R)
    vert_temp[8] = SVector(-R, -R, +R)

    cells_temp =
        [(1, 2, 3, 4), (4, 8, 2, 6), (6, 2, 5, 1), (1, 5, 3, 7), (7, 3, 8, 4), (5, 6, 7, 8)]

    # construct connectivity for cube composed of the 6 cells above
    conn_temp = P4estTypes.Connectivity{4}(vert_temp, cells_temp)

    # refine the 6 cell cube into a 6*nedge^2 cell cube
    conn = P4estTypes.refine(conn_temp, nedge)

    # collect cell of vertex data of the refinement
    verts = GC.@preserve conn collect(P4estTypes.unsafe_vertices(conn))
    vertices = [SVector(verts[i][1], verts[i][2], verts[i][3]) for i = 1:length(verts)]
    cells = GC.@preserve conn map.(x -> x + 1, P4estTypes.unsafe_trees(conn))

    return CoarseGrid{typeof(conn),typeof(vertices),typeof(cells)}(conn, vertices, cells)
end

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
