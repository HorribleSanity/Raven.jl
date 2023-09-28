abstract type AbstractCoarseGrid end

struct MeshImportCoarseGrid{C,V,L,W,U,M} <: AbstractCoarseGrid
    connectivity::C
    vertices::V
    cells::L
    warp::W
    unwarp::U
    MeshImport::AbstractMeshImport
end
connectivity(g::MeshImportCoarseGrid) = g.connectivity
vertices(g::MeshImportCoarseGrid) = g.vertices
cells(g::MeshImportCoarseGrid) = g.cells
meshimport(g::MeshImportCoarseGrid) = g.MeshImport

function coarsegrid(meshfilename::String, warp = identity, unwarp = identity)
    meshimport = abaqusmeshimport(meshfilename)
    cg_temp = coarsegrid(meshimport.nodes, meshimport.connectivity)
    vertices = Raven.vertices(cg_temp)
    cells = Raven.cells(cg_temp)
    if length(cells[begin]) == 4
        conn = P4estTypes.Connectivity{4}(vertices, cells)
    elseif length(cells[begin]) == 8
        conn = P4estTypes.Connectivity{8}(vertices, cells)
    end
    C, V, L, W, U, M = typeof.([conn, vertices, cells, warp, unwarp, meshimport])
    return MeshImportCoarseGrid{C,V,L,W,U,M}(
        conn,
        vertices,
        cells,
        warp,
        unwarp,
        meshimport,
    )
end

struct CoarseGrid{C,V,L,W,U} <: AbstractCoarseGrid
    connectivity::C
    vertices::V
    cells::L
    warp::W
    unwarp::U
end

warp(::AbstractCoarseGrid) = identity
unwarp(::AbstractCoarseGrid) = identity

connectivity(g::CoarseGrid) = g.connectivity
vertices(g::CoarseGrid) = g.vertices
cells(g::CoarseGrid) = g.cells
warp(g::CoarseGrid) = g.warp
unwarp(g::CoarseGrid) = g.unwarp

function coarsegrid(
    vertices,
    cells::AbstractVector{NTuple{X,T}},
    warp = identity,
    unwarp = identity,
) where {X,T}

    conn = P4estTypes.Connectivity{X}(vertices, cells)
    C, V, L, W, U = typeof.([conn, vertices, cells, warp, unwarp])
    return CoarseGrid{C,V,L,W,U}(conn, vertices, cells, warp, unwarp)
end

function cubeshellgrid(R::Real, r::Real)
    @assert R > r "R (outer radius) must be greater that r (inner radius)"
    vertices = zeros(SVector{3,Float64}, 16)

    vertices[1] = SVector(+R, +R, -R)
    vertices[2] = SVector(+R, -R, -R)
    vertices[3] = SVector(+R, +R, +R)
    vertices[4] = SVector(+R, -R, +R)
    vertices[5] = SVector(-R, +R, -R)
    vertices[6] = SVector(-R, -R, -R)
    vertices[7] = SVector(-R, +R, +R)
    vertices[8] = SVector(-R, -R, +R)

    vertices[9] = SVector(+r, +r, -r)
    vertices[10] = SVector(+r, -r, -r)
    vertices[11] = SVector(+r, +r, +r)
    vertices[12] = SVector(+r, -r, +r)
    vertices[13] = SVector(-r, +r, -r)
    vertices[14] = SVector(-r, -r, -r)
    vertices[15] = SVector(-r, +r, +r)
    vertices[16] = SVector(-r, -r, +r)


    cells = [
        (1, 2, 3, 4, 9, 10, 11, 12),
        (2, 6, 4, 8, 10, 14, 12, 16),
        (6, 5, 8, 7, 14, 13, 16, 15),
        (5, 1, 7, 3, 13, 9, 15, 11),
        (3, 4, 7, 8, 11, 12, 15, 16),
        (1, 2, 5, 6, 9, 10, 13, 14),
    ]

    function cubespherewarp(point)
        # Put the points in reverse magnitude order
        p = sortperm(abs.(point))
        point = point[p]

        # Convert to angles
        ξ = π * point[2] / 4point[3]
        η = π * point[1] / 4point[3]

        # Compute the ratios
        y_x = tan(ξ)
        z_x = tan(η)

        # Compute the new points
        x = point[3] / hypot(1, y_x, z_x)
        y = x * y_x
        z = x * z_x

        # Compute the new points and unpermute
        point = SVector(z, y, x)[sortperm(p)]
        return point
    end

    function cubesphereunwarp(point)
        # Put the points in reverse magnitude order
        p = sortperm(abs.(point))
        point = point[p]

        # Convert to angles
        ξ = 4atan(point[2] / point[3]) / π
        η = 4atan(point[1] / point[3]) / π
        R = sign(point[3]) * hypot(point...)

        x = R
        y = R * ξ
        z = R * η
        # Compute the new points and unpermute
        point = SVector(z, y, x)[sortperm(p)]
        return point
    end

    return coarsegrid(vertices, cells, cubespherewarp, cubesphereunwarp)
end


"""
    function cubeshell2dgrid(R::Real)

    This function will construct the CoarseGrid of a cube shell of radius R. 
    A cube shell is a 2D connectivity.
"""
function cubeshell2dgrid(R::Real)
    vertices = zeros(SVector{3,Float64}, 8)

    vertices[1] = SVector(+R, +R, -R)
    vertices[2] = SVector(+R, -R, -R)
    vertices[3] = SVector(+R, +R, +R)
    vertices[4] = SVector(+R, -R, +R)
    vertices[5] = SVector(-R, +R, -R)
    vertices[6] = SVector(-R, -R, -R)
    vertices[7] = SVector(-R, +R, +R)
    vertices[8] = SVector(-R, -R, +R)

    cells =
        [(1, 2, 3, 4), (4, 8, 2, 6), (6, 2, 5, 1), (1, 5, 3, 7), (7, 3, 8, 4), (5, 6, 7, 8)]

    function cubespherewarp(point)
        # Put the points in reverse magnitude order
        p = sortperm(abs.(point))
        point = point[p]

        # Convert to angles
        ξ = π * point[2] / 4point[3]
        η = π * point[1] / 4point[3]

        # Compute the ratios
        y_x = tan(ξ)
        z_x = tan(η)

        # Compute the new points
        x = point[3] / hypot(1, y_x, z_x)
        y = x * y_x
        z = x * z_x

        # Compute the new points and unpermute
        point = SVector(z, y, x)[sortperm(p)]
        return point
    end

    function cubesphereunwarp(point)
        # Put the points in reverse magnitude order
        p = sortperm(abs.(point))
        point = point[p]

        # Convert to angles
        ξ = 4atan(point[2] / point[3]) / π
        η = 4atan(point[1] / point[3]) / π
        R = sign(point[3]) * hypot(point...)

        x = R
        y = R * ξ
        z = R * η
        # Compute the new points and unpermute
        point = SVector(z, y, x)[sortperm(p)]
        return point
    end

    return coarsegrid(vertices, cells, cubespherewarp, cubesphereunwarp)
end

struct BrickGrid{T,C,D} <: AbstractCoarseGrid
    connectivity::C
    coordinates::D
end

connectivity(g::BrickGrid) = g.connectivity
coordinates(g::BrickGrid) = g.coordinates
function vertices(g::BrickGrid{T}) where {T}
    conn = connectivity(g)
    coords = coordinates(g)
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
    return verts
end
function cells(g::BrickGrid)
    conn = connectivity(g)
    GC.@preserve conn map.(x -> x + 1, P4estTypes.unsafe_trees(conn))
end

function brick(T::Type, coordinates, p)
    n = length.(coordinates) .- 0x1
    connectivity = P4estTypes.brick(n, p)

    return BrickGrid{T,typeof(connectivity),typeof(coordinates)}(connectivity, coordinates)
end

function brick(coordinates::Tuple{<:Any,<:Any}, p::Tuple{Bool,Bool} = (false, false))
    T = promote_type(eltype.(coordinates)...)
    return brick(T, coordinates, p)
end

function brick(
    coordinates::Tuple{<:Any,<:Any,<:Any},
    p::Tuple{Bool,Bool,Bool} = (false, false, false),
)
    T = promote_type(eltype.(coordinates)...)
    return brick(T, coordinates, p)
end

function brick(T::Type, n::Tuple{Integer,Integer}, p::Tuple{Bool,Bool} = (false, false))
    coordinates = (zero(T):n[1], zero(T):n[2])
    return brick(T, coordinates, p)
end

function brick(
    T::Type,
    n::Tuple{Integer,Integer,Integer},
    p::Tuple{Bool,Bool,Bool} = (false, false, false),
)
    coordinates = (zero(T):n[1], zero(T):n[2], zero(T):n[3])
    return brick(T, coordinates, p)
end

function brick(n::Tuple{Integer,Integer}, p::Tuple{Bool,Bool} = (false, false))
    return brick(Float64, n, p)
end

function brick(
    n::Tuple{Integer,Integer,Integer},
    p::Tuple{Bool,Bool,Bool} = (false, false, false),
)
    return brick(Float64, n, p)
end

brick(a::AbstractArray, b::AbstractArray, p::Bool = false, q::Bool = false) =
    brick((a, b), (p, q))
brick(l::Integer, m::Integer, p::Bool = false, q::Bool = false) =
    brick(Float64, (l, m), (p, q))
brick(T::Type, l::Integer, m::Integer, p::Bool = false, q::Bool = false) =
    brick(T, (l, m), (p, q))

function brick(
    a::AbstractArray,
    b::AbstractArray,
    c::AbstractArray,
    p::Bool = false,
    q::Bool = false,
    r::Bool = false,
)
    return brick((a, b, c), (p, q, r))
end

function brick(
    l::Integer,
    m::Integer,
    n::Integer,
    p::Bool = false,
    q::Bool = false,
    r::Bool = false,
)
    return brick(Float64, (l, m, n), (p, q, r))
end

function brick(
    T::Type,
    l::Integer,
    m::Integer,
    n::Integer,
    p::Bool = false,
    q::Bool = false,
    r::Bool = false,
)
    return brick(T, (l, m, n), (p, q, r))
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
