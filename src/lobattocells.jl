function lobattooperators_1d(::Type{T}, M) where {T}
    oldprecision = precision(BigFloat)

    # Increase precision of the type used to compute the 1D operators to help
    # ensure any symmetries.  This is not thread safe so an alternative would
    # be to use ArbNumerics.jl which keeps its precision in the type.
    setprecision(BigFloat, 2^(max(8, ceil(Int, log2(precision(T))) + 2)))

    points, weights = legendregausslobatto(BigFloat, M)
    gausspoints, _ = legendregauss(BigFloat, M)
    derivative = spectralderivative(points)
    weightedderivative = weights .* derivative
    skewweightedderivative = (weightedderivative - weightedderivative') / 2
    equallyspacedpoints = range(-one(BigFloat), stop = one(BigFloat), length = M)
    toequallyspaced = spectralinterpolation(points, equallyspacedpoints)
    tolowerhalf = spectralinterpolation(points, (points .- 1) ./ 2)
    toupperhalf = spectralinterpolation(points, (points .+ 1) ./ 2)
    togauss = spectralinterpolation(points, gausspoints)
    tolobatto = spectralinterpolation(points, points)
    toboundary = spectralinterpolation(points, [-1, 1])

    setprecision(oldprecision)

    points = Array{T}(points)
    weights = Array{T}(weights)
    derivative = Array{T}(derivative)
    weightedderivative = Array{T}(weightedderivative)
    skewweightedderivative = Array{T}(skewweightedderivative)
    toequallyspaced = Array{T}(toequallyspaced)
    tohalves = (Array{T}(tolowerhalf), Array{T}(toupperhalf))
    togauss = Array{T}(togauss)
    tolobatto = Array{T}(tolobatto)
    toboundary = Array{T}(toboundary)

    return (;
        points,
        weights,
        derivative,
        weightedderivative,
        skewweightedderivative,
        toequallyspaced,
        tohalves,
        togauss,
        tolobatto,
        toboundary,
    )
end

struct LobattoCell{T,A,N,S,O,P,D,WD,SWD,M,FM,E,H,TG,TL,TB} <: AbstractCell{T,A,N}
    size::S
    points_1d::O
    weights_1d::O
    points::P
    derivatives::D
    weightedderivatives::WD
    skewweightedderivatives::SWD
    mass::M
    facemass::FM
    toequallyspaced::E
    tohalves_1d::H
    togauss::TG
    tolobatto::TL
    toboundary::TB
end

function Base.similar(::LobattoCell{T,A}, dims::Dims) where {T,A}
    return LobattoCell{T,A}(dims...)
end

function Base.show(io::IO, cell::LobattoCell{T,A}) where {T,A}
    print(io, "LobattoCell{")
    Base.show(io, T)
    print(io, ", ")
    Base.show(io, A)
    print(io, "}$(size(cell))")
end

function Base.showarg(io::IO, ::LobattoCell{T,A,N}, toplevel) where {T,A,N}
    !toplevel && print(io, "::")
    print(io, "LobattoCell{", T, ",", A, ",", N, "}")
    return
end

function Base.summary(io::IO, cell::LobattoCell{T,A,N}) where {T,A,N}
    d = Base.dims2string(size(cell))
    print(io, "$d LobattoCell{", T, ",", A, ",", N, "}")
end

function LobattoCell{T,A}(m) where {T,A}
    o = adapt(A, lobattooperators_1d(T, m))
    points_1d = (o.points,)
    weights_1d = (o.weights,)

    points = vec(SVector.(points_1d...))
    derivatives = (Kron((o.derivative,)),)
    weightedderivatives = (Kron((o.weightedderivative,)),)
    skewweightedderivatives = (Kron((o.skewweightedderivative,)),)
    mass = Diagonal(vec(.*(weights_1d...)))
    facemass = adapt(A, Diagonal([T(1), T(1)]))
    toequallyspaced = Kron((o.toequallyspaced,))
    tohalves_1d = ((o.tohalves[1], o.tohalves[2]),)
    togauss = Kron((o.togauss,))
    tolobatto = Kron((o.tolobatto,))
    toboundary = Kron((o.toboundary,))

    args = (
        (m,),
        points_1d,
        weights_1d,
        points,
        derivatives,
        weightedderivatives,
        skewweightedderivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
        togauss,
        tolobatto,
        toboundary,
    )
    LobattoCell{T,A,1,typeof(args[1]),typeof.(args[3:end])...}(args...)
end

function LobattoCell{T,A}(m1, m2) where {T,A}
    o = adapt(A, (lobattooperators_1d(T, m1), lobattooperators_1d(T, m2)))

    points_1d = (reshape(o[1].points, (m1, 1)), reshape(o[2].points, (1, m2)))
    weights_1d = (reshape(o[1].weights, (m1, 1)), reshape(o[2].weights, (1, m2)))
    points = vec(SVector.(points_1d...))
    derivatives = (Kron((Eye{T}(m2), o[1].derivative)), Kron((o[2].derivative, Eye{T}(m1))))
    weightedderivatives = (
        Kron((Eye{T}(m2), o[1].weightedderivative)),
        Kron((o[2].weightedderivative, Eye{T}(m1))),
    )
    skewweightedderivatives = (
        Kron((Eye{T}(m2), o[1].skewweightedderivative)),
        Kron((o[2].skewweightedderivative, Eye{T}(m1))),
    )
    mass = Diagonal(vec(.*(weights_1d...)))
    ω1, ω2 = weights_1d
    facemass = Diagonal(vcat(repeat(vec(ω2), 2), repeat(vec(ω1), 2)))
    toequallyspaced = Kron((o[2].toequallyspaced, o[1].toequallyspaced))
    tohalves_1d =
        ((o[1].tohalves[1], o[1].tohalves[2]), (o[2].tohalves[1], o[2].tohalves[2]))
    togauss = Kron((o[2].togauss, o[1].togauss))
    tolobatto = Kron((o[2].tolobatto, o[1].tolobatto))
    toboundary = Kron((o[2].toboundary, o[1].toboundary))


    args = (
        (m1, m2),
        points_1d,
        weights_1d,
        points,
        derivatives,
        weightedderivatives,
        skewweightedderivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
        togauss,
        tolobatto,
        toboundary,
    )
    LobattoCell{T,A,2,typeof(args[1]),typeof.(args[3:end])...}(args...)
end

function LobattoCell{T,A}(m1, m2, m3) where {T,A}
    o = adapt(
        A,
        (
            lobattooperators_1d(T, m1),
            lobattooperators_1d(T, m2),
            lobattooperators_1d(T, m3),
        ),
    )

    points_1d = (
        reshape(o[1].points, (m1, 1, 1)),
        reshape(o[2].points, (1, m2, 1)),
        reshape(o[3].points, (1, 1, m3)),
    )
    weights_1d = (
        reshape(o[1].weights, (m1, 1, 1)),
        reshape(o[2].weights, (1, m2, 1)),
        reshape(o[3].weights, (1, 1, m3)),
    )
    points = vec(SVector.(points_1d...))
    derivatives = (
        Kron((Eye{T}(m3), Eye{T}(m2), o[1].derivative)),
        Kron((Eye{T}(m3), o[2].derivative, Eye{T}(m1))),
        Kron((o[3].derivative, Eye{T}(m2), Eye{T}(m1))),
    )
    weightedderivatives = (
        Kron((Eye{T}(m3), Eye{T}(m2), o[1].weightedderivative)),
        Kron((Eye{T}(m3), o[2].weightedderivative, Eye{T}(m1))),
        Kron((o[3].weightedderivative, Eye{T}(m2), Eye{T}(m1))),
    )
    skewweightedderivatives = (
        Kron((Eye{T}(m3), Eye{T}(m2), o[1].skewweightedderivative)),
        Kron((Eye{T}(m3), o[2].skewweightedderivative, Eye{T}(m1))),
        Kron((o[3].skewweightedderivative, Eye{T}(m2), Eye{T}(m1))),
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
    togauss = Kron((o[3].togauss, o[2].togauss, o[1].togauss))
    tolobatto = Kron((o[3].tolobatto, o[2].tolobatto, o[1].tolobatto))
    toboundary = Kron((o[3].toboundary, o[2].toboundary, o[1].toboundary))

    args = (
        (m1, m2, m3),
        points_1d,
        weights_1d,
        points,
        derivatives,
        weightedderivatives,
        skewweightedderivatives,
        mass,
        facemass,
        toequallyspaced,
        tohalves_1d,
        togauss,
        tolobatto,
        toboundary,
    )
    LobattoCell{T,A,3,typeof(args[1]),typeof.(args[3:end])...}(args...)
end

LobattoCell{T}(args...) where {T} = LobattoCell{T,Array}(args...)
LobattoCell(args...) = LobattoCell{Float64}(args...)

function Adapt.adapt_structure(to, cell::LobattoCell{T,A,N}) where {T,A,N}
    names = fieldnames(LobattoCell)
    args = ntuple(j -> adapt(to, getfield(cell, names[j])), length(names))
    B = arraytype(to)

    LobattoCell{T,B,N,typeof(args[1]),typeof.(args[3:end])...}(args...)
end

const LobattoLine{T,A} = LobattoCell{T,A,1}
const LobattoQuad{T,A} = LobattoCell{T,A,2}
const LobattoHex{T,A} = LobattoCell{T,A,3}

Base.size(cell::LobattoCell) = cell.size
points_1d(cell::LobattoCell) = cell.points_1d
weights_1d(cell::LobattoCell) = cell.weights_1d
points(cell::LobattoCell) = cell.points
derivatives(cell::LobattoCell) = cell.derivatives
weightedderivatives(cell::LobattoCell) = cell.weightedderivatives
skewweightedderivatives(cell::LobattoCell) = cell.skewweightedderivatives
function derivatives_1d(cell::LobattoCell)
    N = ndims(cell)
    ntuple(i -> cell.derivatives[i].args[N-i+1], Val(N))
end
function weightedderivatives_1d(cell::LobattoCell)
    N = ndims(cell)
    ntuple(i -> cell.weightedderivatives[i].args[N-i+1], Val(N))
end
function skewweightedderivatives_1d(cell::LobattoCell)
    N = ndims(cell)
    ntuple(i -> cell.skewweightedderivatives[i].args[N-i+1], Val(N))
end
mass(cell::LobattoCell) = cell.mass
facemass(cell::LobattoCell) = cell.facemass
toequallyspaced(cell::LobattoCell) = cell.toequallyspaced
tohalves_1d(cell::LobattoCell) = cell.tohalves_1d
degrees(cell::LobattoCell) = size(cell) .- 1
togauss(cell::LobattoCell) = cell.togauss
tolobatto(cell::LobattoCell) = cell.tolobatto
toboundary(cell::LobattoCell) = cell.toboundary
togauss_1d(cell::LobattoCell) = reverse(cell.togauss.args)
tolobatto_1d(cell::LobattoCell) = reverse(cell.tolobatto.args)
toboundary_1d(cell::LobattoCell) = reverse(cell.toboundary.args)

faceoffsets(::LobattoLine) = (0, 1, 2)
function faceoffsets(cell::LobattoQuad)
    Nf = (size(cell, 2), size(cell, 2), size(cell, 1), size(cell, 1))
    return cumsum((0, Nf...))
end

function faceoffsets(cell::LobattoHex)
    Nf = (
        size(cell, 2) * size(cell, 3),
        size(cell, 2) * size(cell, 3),
        size(cell, 1) * size(cell, 3),
        size(cell, 1) * size(cell, 3),
        size(cell, 1) * size(cell, 2),
        size(cell, 1) * size(cell, 2),
    )
    return cumsum((0, Nf...))
end

@kernel function quadpoints!(
    points,
    ri,
    si,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{Q},
) where {I,J,Q}
    i, j, q1 = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, Q)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && q1 == 1
                rl[i] = ri[i]
            end

            if i == 1 && q1 == 1
                sl[j] = si[j]
            end

            if i ≤ 2 && j ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                treecoords[i, j, q1] = coarsegridvertices[vids[i+2*(j-1)]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN
            cs = T(iy) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = cr + h * (rl[i] + 1)
            s = cs + h * (sl[j] + 1)

            w1 = (1 - r) * (1 - s)
            w2 = r * (1 - s)
            w3 = (1 - r) * s
            w4 = r * s

            c1 = treecoords[1, 1, q1]
            c2 = treecoords[2, 1, q1]
            c3 = treecoords[1, 2, q1]
            c4 = treecoords[2, 2, q1]

            points[i, j, q] = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4
        end
    end
end

function materializepoints(
    referencecell::LobattoQuad,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))
    Q = max(512 ÷ prod(length.(r)), 1)

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)

    kernel! = quadpoints!(backend, (length.(r)..., Q))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...,
        Val(Q);
        ndrange = size(points),
    )

    return points
end

function materializepoints(
    referencecell::LobattoQuad,
    coarsegridcells,
    coarsegridvertices,
    interpolation_degree,
    faceinterpolation,
    quadranttointerpolation,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))

    FT = floattype(referencecell)
    AT = arraytype(referencecell)
    N = (interpolation_degree + 1, interpolation_degree + 1)
    interp_r = vec.(points_1d(LobattoCell{FT,AT}(N...)))

    Q = max(512 ÷ prod(length.(r)), 1)

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)

    kernel! = curvedquadpoints!(backend, (length.(r)..., Q))
    kernel!(
        points,
        r...,
        interp_r...,
        coarsegridcells,
        coarsegridvertices,
        interpolation_degree,
        faceinterpolation,
        quadranttointerpolation,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...,
        Val(Q);
        ndrange = size(points),
    )

    return points
end

@inline function interp(t, n, data, offset, degree)
    x = zero(t)
    y = zero(t)
    t = 2 * t - 1
    for i = 1:(degree+1)
        li = one(t)
        for j = 1:(degree+1)
            if i != j
                li *= (t - n[j]) / (n[i] - n[j])
            end
        end
        x += data[offset+i-1, 1] * li
        y += data[offset+i-1, 2] * li
    end
    return x, y
end

@inline function bary_interp(t, n, data, offset, degree)
    x = zero(t)
    y = zero(t)
    denom = zero(t)
    t = 2 * t - 1
    w = ones(typeof(t), degree + 1)

    for i = 1:(degree+1)
        for j = 1:(degree+1)
            if i ≠ j
                w[i] *= (n[i] - n[j])
            end
        end
        w[i] = 1 / w[i]
    end

    for i = 1:(degree+1)
        if i != 1 || i != degree + 1
            x += w[i] * data[offset+i-1, 1] / (t - n[i])
            y += w[i] * data[offset+i-1, 2] / (t - n[i])
            denom += w[i] / (t - n[i])
        else
            x += data[offset+i-1, 1]
            y += data[offset+i-1, 2]
        end
    end

    return x / denom, y / denom
end

@inline function interp(t, s, n_t, n_s, data, offset, degree)
    x = zero(t)
    y = zero(t)
    z = zero(t)

    for i = 1:(degree+1)
        for j = 1:(degree+1)
            li_t = one(t)
            li_s = one(s)
            for k = 1:(degree+1)
                if i != k
                    li_t *= (t - n_t[k]) / (n_t[i] - n_t[k])
                end
            end

            for k = 1:(degree+1)
                if j != k
                    li_s *= (s - n_s[k]) / (n_s[j] - n_s[k])
                end
            end
            x += data[offset+(j-1)*(degree+1)+i-1, 1] * li_t * li_s
            y += data[offset+(j-1)*(degree+1)+i-1, 2] * li_t * li_s
            z += data[offset+(j-1)*(degree+1)+i-1, 3] * li_t * li_s
        end
    end
    return x, y, z
end

@kernel function curvedquadpoints!(
    points,
    ri,
    si,
    interp_r,
    interp_s,
    coarsegridcells,
    coarsegridvertices,
    interpolation_degree,
    faceinterpolation,
    quadranttointerpolation,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{Q},
) where {I,J,Q}
    i, j, q1 = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(eltype(points))
    treecoords = @localmem eltype(points) (2, 2, Q)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && q1 == 1
                rl[i] = ri[i]
            end

            if i == 1 && q1 == 1
                sl[j] = si[j]
            end

            if i ≤ 2 && j ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                treecoords[i, j, q1] = coarsegridvertices[vids[i+2*(j-1)]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN
            cs = T(iy) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = cr + h * (rl[i] + 1)
            s = cs + h * (sl[j] + 1)

            interp_idx1 = quadranttointerpolation[treeid, 1]
            interp_idx2 = quadranttointerpolation[treeid, 2]
            interp_idx3 = quadranttointerpolation[treeid, 3]
            interp_idx4 = quadranttointerpolation[treeid, 4]

            c1 = treecoords[1, 1, q1]
            c2 = treecoords[2, 1, q1]
            c3 = treecoords[1, 2, q1]
            c4 = treecoords[2, 2, q1]

            if interp_idx1 != 0
                f1x, f1y = interp(
                    r,
                    interp_r,
                    faceinterpolation,
                    interp_idx1,
                    interpolation_degree,
                )
            else
                f1x, f1y = c1 .+ (r * (c2 - c1))
            end

            if interp_idx2 != 0
                f2x, f2y = interp(
                    s,
                    interp_s,
                    faceinterpolation,
                    interp_idx2,
                    interpolation_degree,
                )
            else
                f2x, f2y = c2 .+ (s * (c4 - c2))
            end

            if interp_idx3 != 0
                f3x, f3y = interp(
                    r,
                    interp_r,
                    faceinterpolation,
                    interp_idx3,
                    interpolation_degree,
                )
            else
                f3x, f3y = c3 .+ (r * (c4 - c3))
            end

            if interp_idx4 != 0
                f4x, f4y = interp(
                    s,
                    interp_s,
                    faceinterpolation,
                    interp_idx4,
                    interpolation_degree,
                )
            else
                f4x, f4y = c1 .+ (s * (c3 - c1))
            end

            x = (1 - s) * f1x + s * f3x + (1 - r) * f4x + r * f2x
            -(1 - s) * (1 - r) * c1[1] - (1 - s) * r * c3[1]
            -s * (1 - r) * c2[1] - s * r * c4[1]

            y = (1 - s) * f1y + s * f3y + (1 - r) * f4y + r * f2y
            -(1 - s) * (1 - r) * c1[2] - (1 - s) * r * c3[2]
            -s * (1 - r) * c2[2] - s * r * c4[2]

            points[i, j, q] = (x / 2, y / 2)
        end
    end
end

@kernel function linebrickpoints!(
    points,
    ri,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{Q},
) where {I,Q}
    i, q1 = @index(Local, NTuple)
    _, q = @index(Global, NTuple)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, Q)
    rl = @localmem eltype(ri) (I,)

    @inbounds begin
        if q ≤ numberofquadrants
            if q1 == 1
                rl[i] = ri[i]
            end

            if i ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                treecoords[i, q1] = coarsegridvertices[vids[i]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = muladd(h, (rl[i] + 1), cr)

            c1 = treecoords[1, q1]
            c2 = treecoords[2, q1]

            dx = c2[1] - c1[1]

            points[i, q] = SVector(muladd(dx, r, c1[1]))
        end
    end
end

function materializebrickpoints(
    referencecell::LobattoLine,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))
    Q = max(512 ÷ prod(length.(r)), 1)

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)

    kernel! = linebrickpoints!(backend, (length.(r)..., Q))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...,
        Val(Q);
        ndrange = size(points),
    )

    return points
end

@kernel function quadbrickpoints!(
    points,
    ri,
    si,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{Q},
) where {I,J,Q}
    i, j, q1 = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, Q)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && q1 == 1
                rl[i] = ri[i]
            end

            if i == 1 && q1 == 1
                sl[j] = si[j]
            end

            if i ≤ 2 && j ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                treecoords[i, j, q1] = coarsegridvertices[vids[i+2*(j-1)]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN
            cs = T(iy) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = muladd(h, (rl[i] + 1), cr)
            s = muladd(h, (sl[j] + 1), cs)

            c1 = treecoords[1, 1, q1]
            c2 = treecoords[2, 1, q1]
            c3 = treecoords[1, 2, q1]

            dx = c2[1] - c1[1]
            dy = c3[2] - c1[2]

            points[i, j, q] = SVector(muladd(dx, r, c1[1]), muladd(dy, s, c1[2]))
        end
    end
end

function materializebrickpoints(
    referencecell::LobattoQuad,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))
    Q = max(512 ÷ prod(length.(r)), 1)

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)

    kernel! = quadbrickpoints!(backend, (length.(r)..., Q))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...,
        Val(Q);
        ndrange = size(points),
    )

    return points
end

@kernel function hexpoints!(
    points,
    ri,
    si,
    ti,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{K},
) where {I,J,K}
    i, j, k = @index(Local, NTuple)
    q = @index(Group, Linear)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, 2)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)
    tl = @localmem eltype(ti) (K,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && k == 1
                rl[i] = ri[i]
            end

            if i == 1 && k == 1
                sl[j] = si[j]
            end

            if i == 1 && j == 1
                tl[k] = ti[k]
            end

            if i ≤ 2 && j ≤ 2 && k ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                id = i + 2 * (j - 1) + 4 * (k - 1)
                treecoords[i, j, k] = coarsegridvertices[vids[id]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]
            iz = quadranttocoordinate[q, 3]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN
            cs = T(iy) / P4EST_ROOT_LEN
            ct = T(iz) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = cr + h * (rl[i] + 1)
            s = cs + h * (sl[j] + 1)
            t = ct + h * (tl[k] + 1)

            w1 = (1 - r) * (1 - s) * (1 - t)
            w2 = r * (1 - s) * (1 - t)
            w3 = (1 - r) * s * (1 - t)
            w4 = r * s * (1 - t)
            w5 = (1 - r) * (1 - s) * t
            w6 = r * (1 - s) * t
            w7 = (1 - r) * s * t
            w8 = r * s * t

            points[i, j, k, q] =
                w1 * treecoords[1] +
                w2 * treecoords[2] +
                w3 * treecoords[3] +
                w4 * treecoords[4] +
                w5 * treecoords[5] +
                w6 * treecoords[6] +
                w7 * treecoords[7] +
                w8 * treecoords[8]
        end
    end
end

function materializepoints(
    referencecell::LobattoHex,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)
    kernel! = hexpoints!(backend, length.(r))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...;
        ndrange = size(points),
    )

    return points
end

@kernel function curvedhexpoints!(
    points,
    ri,
    si,
    ti,
    interp_r,
    interp_s,
    interp_t,
    coarsegridcells,
    coarsegridvertices,
    interpolation_degree,
    faceinterpolation,
    quadranttointerpolation,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{K},
) where {I,J,K}
    i, j, k = @index(Local, NTuple)
    q = @index(Group, Linear)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, 2)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)
    tl = @localmem eltype(ti) (K,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && k == 1
                rl[i] = ri[i]
            end

            if i == 1 && k == 1
                sl[j] = si[j]
            end

            if i == 1 && j == 1
                tl[k] = ti[k]
            end

            if i ≤ 2 && j ≤ 2 && k ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                id = i + 2 * (j - 1) + 4 * (k - 1)
                treecoords[i, j, k] = coarsegridvertices[vids[id]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]
            iz = quadranttocoordinate[q, 3]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN
            cs = T(iy) / P4EST_ROOT_LEN
            ct = T(iz) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level))

            r = (2 * cr - 1) + h * (rl[i] + 1)
            s = (2 * cs - 1) + h * (sl[j] + 1)
            t = (2 * ct - 1) + h * (tl[k] + 1)

            p1 = treecoords[1]
            p2 = treecoords[2]
            p3 = treecoords[4]
            p4 = treecoords[3]
            p5 = treecoords[5]
            p6 = treecoords[6]
            p7 = treecoords[8]
            p8 = treecoords[7]

            interp_idx1 = quadranttointerpolation[treeid, 1]
            interp_idx2 = quadranttointerpolation[treeid, 2]
            interp_idx3 = quadranttointerpolation[treeid, 3]
            interp_idx4 = quadranttointerpolation[treeid, 4]
            interp_idx5 = quadranttointerpolation[treeid, 5]
            interp_idx6 = quadranttointerpolation[treeid, 6]

            # Below f1x corresponds to the x component of the interpolation onto the hex face 1. 
            # f1e1x: face 1 edge 1 x coordinate
            # HOHQMesh data is in rhr vertex order
            #
            #        8-----------7
            #        |\           \
            #        | \           \
            #        |  \           \
            #        |   5-----------6
            #        |   |           |
            #        4   |       3   |
            #         \  |           |
            #          \ |           |
            #           \|           |
            #            1-----------2
            #  t  s
            #  | /
            #  |/
            #  --->r


            if interp_idx1 != 0
                f1x, f1y, f1z = interp(
                    r,
                    t,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx1,
                    interpolation_degree,
                )

                # edge interpolation on face 1
                f1e1x, f1e1y, f1e1z = interp(
                    r,
                    -1,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx1,
                    interpolation_degree,
                )
                f1e2x, f1e2y, f1e2z = interp(
                    1,
                    t,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx1,
                    interpolation_degree,
                )
                f1e3x, f1e3y, f1e3z = interp(
                    r,
                    1,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx1,
                    interpolation_degree,
                )
                f1e4x, f1e4y, f1e4z = interp(
                    -1,
                    t,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx1,
                    interpolation_degree,
                )
            else
                f1x, f1y, f1z =
                    (
                        p1 * (1 - r) * (1 - t) +
                        p2 * (r + 1) * (1 - t) +
                        p5 * (1 - r) * (t + 1) +
                        p6 * (r + 1) * (t + 1)
                    ) / 4

                f1e1x, f1e1y, f1e1z = (p1 * (1 - r) + p2 * (r + 1)) / 2 #edge 1
                f1e2x, f1e2y, f1e2z = (p2 * (1 - t) + p6 * (t + 1)) / 2 #edge 2
                f1e3x, f1e3y, f1e3z = (p5 * (1 - r) + p6 * (r + 1)) / 2 #edge 3
                f1e4x, f1e4y, f1e4z = (p1 * (1 - t) + p5 * (t + 1)) / 2 #edge 4
            end

            if interp_idx2 != 0
                f2x, f2y, f2z = interp(
                    r,
                    t,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx2,
                    interpolation_degree,
                )

                # edge interpolation on face 2
                f2e1x, f2e1y, f2e1z = interp(
                    r,
                    -1,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx2,
                    interpolation_degree,
                )
                f2e2x, f2e2y, f2e2z = interp(
                    1,
                    t,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx2,
                    interpolation_degree,
                )
                f2e3x, f2e3y, f2e3z = interp(
                    r,
                    1,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx2,
                    interpolation_degree,
                )
                f2e4x, f2e4y, f2e4z = interp(
                    -1,
                    t,
                    interp_r,
                    interp_t,
                    faceinterpolation,
                    interp_idx2,
                    interpolation_degree,
                )
            else
                f2x, f2y, f2z =
                    (
                        p4 * (1 - r) * (1 - t) +
                        p3 * (r + 1) * (1 - t) +
                        p8 * (1 - r) * (t + 1) +
                        p7 * (r + 1) * (t + 1)
                    ) / 4

                f2e1x, f2e1y, f2e1z = (p4 * (1 - r) + p3 * (r + 1)) / 2 #edge 5
                f2e2x, f2e2y, f2e2z = (p3 * (1 - t) + p7 * (t + 1)) / 2 #edge 6
                f2e3x, f2e3y, f2e3z = (p8 * (1 - r) + p7 * (r + 1)) / 2 #edge 7
                f2e4x, f2e4y, f2e4z = (p4 * (1 - t) + p8 * (t + 1)) / 2 #edge 8
            end

            if interp_idx3 != 0
                f3x, f3y, f3z = interp(
                    r,
                    s,
                    interp_r,
                    interp_s,
                    faceinterpolation,
                    interp_idx3,
                    interpolation_degree,
                )
            else
                f3x, f3y, f3z =
                    (
                        p1 * (1 - r) * (1 - s) +
                        p2 * (r + 1) * (1 - s) +
                        p4 * (1 - r) * (s + 1) +
                        p3 * (r + 1) * (s + 1)
                    ) / 4
            end

            if interp_idx4 != 0
                f4x, f4y, f4z = interp(
                    s,
                    t,
                    interp_s,
                    interp_t,
                    faceinterpolation,
                    interp_idx4,
                    interpolation_degree,
                )

                # edge interpolation on face 4 
                f4e1x, f4e1y, f4e1z = interp(
                    s,
                    -1,
                    interp_s,
                    interp_t,
                    faceinterpolation,
                    interp_idx4,
                    interpolation_degree,
                )
                f4e2x, f4e2y, f4e2z = interp(
                    s,
                    1,
                    interp_s,
                    interp_t,
                    faceinterpolation,
                    interp_idx4,
                    interpolation_degree,
                )
            else
                f4x, f4y, f4z =
                    (
                        p2 * (1 - s) * (1 - t) +
                        p3 * (s + 1) * (1 - t) +
                        p6 * (1 - s) * (t + 1) +
                        p7 * (s + 1) * (t + 1)
                    ) / 4

                # edge interpolation on face 4 for linear face
                f4e1x, f4e1y, f4e1z = (p2 * (1 - s) + p3 * (s + 1)) / 2 # edge 10
                f4e2x, f4e2y, f4e2z = (p6 * (1 - s) + p7 * (s + 1)) / 2 # edge 11
            end

            if interp_idx5 != 0
                f5x, f5y, f5z = interp(
                    r,
                    s,
                    interp_r,
                    interp_s,
                    faceinterpolation,
                    interp_idx5,
                    interpolation_degree,
                )
            else
                f5x, f5y, f5z =
                    (
                        p5 * (1 - r) * (1 - s) +
                        p6 * (r + 1) * (1 - s) +
                        p8 * (1 - r) * (s + 1) +
                        p7 * (r + 1) * (s + 1)
                    ) / 4
            end

            if interp_idx6 != 0
                f6x, f6y, f6z = interp(
                    s,
                    t,
                    interp_s,
                    interp_t,
                    faceinterpolation,
                    interp_idx6,
                    interpolation_degree,
                )

                f6e1x, f6e1y, f6e1z = interp(
                    s,
                    -1,
                    interp_s,
                    interp_t,
                    faceinterpolation,
                    interp_idx6,
                    interpolation_degree,
                )
                f6e2x, f6e2y, f6e2z = interp(
                    s,
                    1,
                    interp_s,
                    interp_t,
                    faceinterpolation,
                    interp_idx6,
                    interpolation_degree,
                )
            else
                f6x, f6y, f6z =
                    (
                        p1 * (1 - s) * (1 - t) +
                        p4 * (s + 1) * (1 - t) +
                        p5 * (1 - s) * (t + 1) +
                        p8 * (s + 1) * (t + 1)
                    ) / 4

                f6e1x, f6e1y, f6e1z = (p1 * (1 - s) + p4 * (s + 1)) / 2 # edge 9
                f6e2x, f6e2y, f6e2z = (p5 * (1 - s) + p8 * (s + 1)) / 2 # edge 12
            end

            x =
                (
                    (1 - r) * f6x +
                    (1 + r) * f4x +
                    (1 - s) * f1x +
                    (1 + s) * f2x +
                    (1 - t) * f3x +
                    (1 + t) * f5x
                ) / 2
            y =
                (
                    (1 - r) * f6y +
                    (1 + r) * f4y +
                    (1 - s) * f1y +
                    (1 + s) * f2y +
                    (1 - t) * f3y +
                    (1 + t) * f5y
                ) / 2
            z =
                (
                    (1 - r) * f6z +
                    (1 + r) * f4z +
                    (1 - s) * f1z +
                    (1 + s) * f2z +
                    (1 - t) * f3z +
                    (1 + t) * f5z
                ) / 2

            x +=
                (
                    p1[1] * (1 - r) * (1 - s) * (1 - t) +
                    p2[1] * (1 + r) * (1 - s) * (1 - t) +
                    p3[1] * (1 + r) * (1 + s) * (1 - t) +
                    p4[1] * (1 - r) * (1 + s) * (1 - t) +
                    p5[1] * (1 - r) * (1 - s) * (1 + t) +
                    p6[1] * (1 + r) * (1 - s) * (1 + t) +
                    p7[1] * (1 + r) * (1 + s) * (1 + t) +
                    p8[1] * (1 - r) * (1 + s) * (1 + t)
                ) / 8

            y +=
                (
                    p1[2] * (1 - r) * (1 - s) * (1 - t) +
                    p2[2] * (1 + r) * (1 - s) * (1 - t) +
                    p3[2] * (1 + r) * (1 + s) * (1 - t) +
                    p4[2] * (1 - r) * (1 + s) * (1 - t) +
                    p5[2] * (1 - r) * (1 - s) * (1 + t) +
                    p6[2] * (1 + r) * (1 - s) * (1 + t) +
                    p7[2] * (1 + r) * (1 + s) * (1 + t) +
                    p8[2] * (1 - r) * (1 + s) * (1 + t)
                ) / 8

            z +=
                (
                    p1[3] * (1 - r) * (1 - s) * (1 - t) +
                    p2[3] * (1 + r) * (1 - s) * (1 - t) +
                    p3[3] * (1 + r) * (1 + s) * (1 - t) +
                    p4[3] * (1 - r) * (1 + s) * (1 - t) +
                    p5[3] * (1 - r) * (1 - s) * (1 + t) +
                    p6[3] * (1 + r) * (1 - s) * (1 + t) +
                    p7[3] * (1 + r) * (1 + s) * (1 + t) +
                    p8[3] * (1 - r) * (1 + s) * (1 + t)
                ) / 8

            x -=
                (
                    (1 - s) * (1 - t) * f1e1x +
                    (1 + r) * (1 - s) * f1e2x +
                    (1 - s) * (1 + t) * f1e3x +
                    (1 - r) * (1 - s) * f1e4x +
                    (1 + s) * (1 - t) * f2e1x +
                    (1 + r) * (1 + s) * f2e2x +
                    (1 + s) * (1 + t) * f2e3x +
                    (1 - r) * (1 + s) * f2e4x +
                    (1 - r) * (1 - t) * f6e1x +
                    (1 + r) * (1 - t) * f4e1x +
                    (1 + r) * (1 + t) * f4e2x +
                    (1 - r) * (1 + t) * f6e2x
                ) / 4

            y -=
                (
                    (1 - s) * (1 - t) * f1e1y +
                    (1 + r) * (1 - s) * f1e2y +
                    (1 - s) * (1 + t) * f1e3y +
                    (1 - r) * (1 - s) * f1e4y +
                    (1 + s) * (1 - t) * f2e1y +
                    (1 + r) * (1 + s) * f2e2y +
                    (1 + s) * (1 + t) * f2e3y +
                    (1 - r) * (1 + s) * f2e4y +
                    (1 - r) * (1 - t) * f6e1y +
                    (1 + r) * (1 - t) * f4e1y +
                    (1 + r) * (1 + t) * f4e2y +
                    (1 - r) * (1 + t) * f6e2y
                ) / 4

            z -=
                (
                    (1 - s) * (1 - t) * f1e1z +
                    (1 + r) * (1 - s) * f1e2z +
                    (1 - s) * (1 + t) * f1e3z +
                    (1 - r) * (1 - s) * f1e4z +
                    (1 + s) * (1 - t) * f2e1z +
                    (1 + r) * (1 + s) * f2e2z +
                    (1 + s) * (1 + t) * f2e3z +
                    (1 - r) * (1 + s) * f2e4z +
                    (1 - r) * (1 - t) * f6e1z +
                    (1 + r) * (1 - t) * f4e1z +
                    (1 + r) * (1 + t) * f4e2z +
                    (1 - r) * (1 + t) * f6e2z
                ) / 4

            points[i, j, k, q] = (x, y, z)
        end
    end
end

function materializepoints(
    referencecell::LobattoHex,
    coarsegridcells,
    coarsegridvertices,
    interpolation_degree,
    faceinterpolation,
    quadranttointerpolation,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))
    FT = floattype(referencecell)
    AT = arraytype(referencecell)
    N = Tuple((interpolation_degree + 1) * ones(Int64, 3))
    interp_r = vec.(points_1d(LobattoCell{FT,AT}(N...)))

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)
    kernel! = curvedhexpoints!(backend, length.(r))
    kernel!(
        points,
        r...,
        interp_r...,
        coarsegridcells,
        coarsegridvertices,
        interpolation_degree,
        faceinterpolation,
        quadranttointerpolation,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...;
        ndrange = size(points),
    )

    return points
end

@kernel function hexbrickpoints!(
    points,
    ri,
    si,
    ti,
    coarsegridcells,
    coarsegridvertices,
    numberofquadrants,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    ::Val{I},
    ::Val{J},
    ::Val{K},
) where {I,J,K}
    i, j, k = @index(Local, NTuple)
    q = @index(Group, Linear)

    @uniform T = eltype(eltype(points))

    treecoords = @localmem eltype(points) (2, 2, 2)
    rl = @localmem eltype(ri) (I,)
    sl = @localmem eltype(si) (J,)
    tl = @localmem eltype(ti) (K,)

    @inbounds begin
        if q ≤ numberofquadrants
            if j == 1 && k == 1
                rl[i] = ri[i]
            end

            if i == 1 && k == 1
                sl[j] = si[j]
            end

            if i == 1 && j == 1
                tl[k] = ti[k]
            end

            if i ≤ 2 && j ≤ 2 && k ≤ 2
                treeid = quadranttotreeid[q]
                vids = coarsegridcells[treeid]
                id = i + 2 * (j - 1) + 4 * (k - 1)
                treecoords[i, j, k] = coarsegridvertices[vids[id]]
            end
        end
    end

    @synchronize

    @inbounds begin
        if q ≤ numberofquadrants
            treeid = quadranttotreeid[q]
            level = quadranttolevel[q]
            ix = quadranttocoordinate[q, 1]
            iy = quadranttocoordinate[q, 2]
            iz = quadranttocoordinate[q, 3]

            P4EST_MAXLEVEL = 30
            P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL

            cr = T(ix) / P4EST_ROOT_LEN
            cs = T(iy) / P4EST_ROOT_LEN
            ct = T(iz) / P4EST_ROOT_LEN

            h = one(T) / (1 << (level + 1))

            r = muladd(h, (rl[i] + 1), cr)
            s = muladd(h, (sl[j] + 1), cs)
            t = muladd(h, (tl[k] + 1), ct)

            c1 = treecoords[1]
            c2 = treecoords[2]
            c3 = treecoords[3]
            c5 = treecoords[5]

            dx = c2[1] - c1[1]
            dy = c3[2] - c1[2]
            dz = c5[3] - c1[3]

            points[i, j, k, q] =
                SVector(muladd(dx, r, c1[1]), muladd(dy, s, c1[2]), muladd(dz, t, c1[3]))
        end
    end
end

function materializebrickpoints(
    referencecell::LobattoHex,
    coarsegridcells,
    coarsegridvertices,
    quadranttolevel,
    quadranttotreeid,
    quadranttocoordinate,
    localnumberofquadrants,
    comm,
)
    r = vec.(points_1d(referencecell))

    IntType = typeof(length(r))
    num_local = IntType(localnumberofquadrants)
    points = GridArray{eltype(coarsegridvertices)}(
        undef,
        arraytype(referencecell),
        (length.(r)..., num_local),
        (length.(r)..., length(quadranttolevel)),
        comm,
        false,
        length(r) + 1,
    )

    backend = get_backend(points)
    kernel! = hexbrickpoints!(backend, length.(r))
    kernel!(
        points,
        r...,
        coarsegridcells,
        coarsegridvertices,
        length(quadranttolevel),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        Val.(length.(r))...;
        ndrange = size(points),
    )

    return points
end

function materializedtoc(cell::LobattoCell, dtoc_degree3_local, dtoc_degree3_global)
    cellsize = size(cell)

    dtoc = zeros(Int, cellsize..., last(size(dtoc_degree3_local)))

    if length(dtoc_degree3_global) > 0
        # Compute the offsets for the cell node numbering
        offsets = zeros(Int, maximum(dtoc_degree3_local) + 1)
        for i in eachindex(IndexCartesian(), dtoc_degree3_local)
            l = dtoc_degree3_local[i]
            I = Tuple(i)
            node = I[1:(end-1)]

            if 3 ∈ node
                # These points are just for orientation
                continue
            end

            # compute the cell dofs for the corner, edge, face or volume identified by node.
            # This is an exclusive count, so the number of dofs in the volume do not include
            # the ones that are also on the faces, edges, or corners.
            ds = ntuple(length(node)) do n
                m = node[n]
                if m == 2
                    d = cellsize[n] - 2
                else
                    d = 1
                end
                return d
            end
            # is this needed???
            if offsets[l+1] == 0
                offsets[l+1] = prod(ds)
            end
        end
        cumsum!(offsets, offsets)

        for i in eachindex(IndexCartesian(), dtoc_degree3_local)
            l = dtoc_degree3_local[i]
            I = Tuple(i)
            node = I[1:(end-1)]
            quad = I[end]

            if 3 ∈ node
                offset_node = map(x -> x == 3 ? 2 : x, node)
                offsets[l] = offsets[dtoc_degree3_local[offset_node..., quad]]
            end
        end

        for i in eachindex(IndexCartesian(), dtoc_degree3_local)
            l = dtoc_degree3_local[i]
            offset = offsets[l]
            I = Tuple(i)
            node = I[1:(end-1)]
            quad = I[end]

            # These points are just for orientation they are not associated
            # with any dofs.
            if 3 ∈ node
                continue
            end

            numtwos = sum(node .== 2)

            dims = ntuple(length(cellsize)) do n
                # We use a StepRange here so that the return type is the same
                # whether or not the dim gets reversed.
                dim =
                    node[n] == 2 ? StepRange(2, Int8(1), cellsize[n] - 1) :
                    node[n] == 1 ? StepRange(1, Int8(1), 1) :
                    StepRange(cellsize[n], Int8(1), cellsize[n])

                return dim
            end

            unrotatedindices = CartesianIndices(dims)

            if numtwos == 0
                for (j, k) in enumerate(unrotatedindices)
                    dtoc[k, quad] = j + offset
                end

            elseif numtwos == 1
                twoindex = findfirst(==(2), node)
                @assert !isnothing(twoindex)
                # edge
                shift = ntuple(m -> m == twoindex ? 1 : 0, length(cellsize))
                # get canonical orientation of the edge
                M = (
                    dtoc_degree3_global[node..., quad]...,
                    dtoc_degree3_global[(node .+ shift)..., quad]...,
                )
                p = orient(Val(2), M)
                edgedims = (cellsize[twoindex] - 2,)

                for (j, ei) in enumerate(CartesianIndices(edgedims))
                    pei = orientindex(p, edgedims, ei)
                    k = unrotatedindices[LinearIndices(edgedims)[pei]]
                    dtoc[k, quad] = j + offset
                end
            elseif numtwos == 2
                # face
                twoindex1 = something(findfirst(==(2), node), 0)
                twoindex2 = something(findnext(==(2), node, twoindex1 + 1), 0)
                @assert twoindex1 != 0
                @assert twoindex2 != 0

                ashift = ntuple(m -> m == twoindex1 ? 1 : 0, length(cellsize))
                bshift = ntuple(m -> m == twoindex2 ? 1 : 0, length(cellsize))
                abshift = ashift .+ bshift
                M = (
                    dtoc_degree3_global[node..., quad]...,
                    dtoc_degree3_global[(node .+ ashift)..., quad]...,
                    dtoc_degree3_global[(node .+ bshift)..., quad]...,
                    dtoc_degree3_global[(node .+ abshift)..., quad]...,
                )
                # get canonical orientation of the edge
                p = orient(Val(4), M)

                facedims = (cellsize[twoindex1] - 2, cellsize[twoindex2] - 2)

                for (j, fi) in enumerate(CartesianIndices(facedims))
                    pfi = orientindex(p, facedims, fi)
                    k = unrotatedindices[LinearIndices(facedims)[pfi]]
                    dtoc[k, quad] = j + offset
                end
            elseif numtwos == 3
                # volume
                for (j, k) in enumerate(unrotatedindices)
                    dtoc[k, quad] = j + offset
                end
            else
                error("Not implemented")
            end
        end
    end

    return dtoc
end

function _indextoface(::Val{2}, i)
    degree3facelinearindices = ((5, 9), (8, 12), (2, 3), (14, 15))
    f = 0

    for (ff, findices) in enumerate(degree3facelinearindices)
        if i ∈ findices
            f = ff
            break
        end
    end

    @assert f != 0

    return f
end


function _indextoface(::Val{3}, i)
    degree3facelinearindices = (
        (21, 25, 37, 41),
        (24, 28, 40, 44),
        (18, 19, 34, 35),
        (30, 31, 46, 47),
        (6, 7, 10, 11),
        (54, 55, 58, 59),
    )

    f = 0

    for (ff, findices) in enumerate(degree3facelinearindices)
        if i ∈ findices
            f = ff
            break
        end
    end

    @assert f != 0

    return f
end


@inline facedims(cell::LobattoQuad) =
    ((size(cell, 2),), (size(cell, 2),), (size(cell, 1),), (size(cell, 1),))

@inline facedims(cell::LobattoHex) = (
    (size(cell, 2), size(cell, 3)),
    (size(cell, 2), size(cell, 3)),
    (size(cell, 1), size(cell, 3)),
    (size(cell, 1), size(cell, 3)),
    (size(cell, 1), size(cell, 2)),
    (size(cell, 1), size(cell, 2)),
)

function materializefacemaps(
    cell::LobattoCell{T,A,N},
    numcells_local,
    ctod_degree3_local,
    dtoc_degree3_local,
    dtoc_degree3_global,
    quadranttolevel,
    quadranttoglobalid,
) where {T,A,N}
    numcells = last(size(dtoc_degree3_local))
    cellindices = LinearIndices(size(cell))

    if cell isa LobattoQuad
        numfaces = 4
        degree3faceindices =
            (((1, 2), (1, 3)), ((4, 2), (4, 3)), ((2, 1), (3, 1)), ((2, 4), (3, 4)))
        cellfacedims = ((size(cell, 2),), (size(cell, 1),))
        facefaceindices = (LinearIndices((size(cell, 2),)), LinearIndices((size(cell, 1),)))
        cellfacedims2 =
            ((size(cell, 2),), (size(cell, 2),), (size(cell, 1),), (size(cell, 1),))
        cellfaceindices = (
            cellindices[1, 1:end],
            cellindices[end, 1:end],
            cellindices[1:end, 1],
            cellindices[1:end, end],
        )
    elseif cell isa LobattoHex
        numfaces = 6
        degree3faceindices = (
            ((1, 2, 2), (1, 3, 2), (1, 2, 3), (1, 3, 3)),
            ((4, 2, 2), (4, 3, 2), (4, 2, 3), (4, 3, 3)),
            ((2, 1, 2), (3, 1, 2), (2, 1, 3), (3, 1, 3)),
            ((2, 4, 2), (3, 4, 2), (2, 4, 3), (3, 4, 3)),
            ((2, 2, 1), (3, 2, 1), (2, 3, 1), (3, 3, 1)),
            ((2, 2, 4), (3, 2, 4), (2, 3, 4), (3, 3, 4)),
        )
        cellfacedims = (
            (size(cell, 2), size(cell, 3)),
            (size(cell, 1), size(cell, 3)),
            (size(cell, 1), size(cell, 2)),
        )
        facefaceindices = (
            LinearIndices((size(cell, 2), size(cell, 3))),
            LinearIndices((size(cell, 1), size(cell, 3))),
            LinearIndices((size(cell, 1), size(cell, 2))),
        )
        cellfacedims2 = (
            (size(cell, 2), size(cell, 3)),
            (size(cell, 2), size(cell, 3)),
            (size(cell, 1), size(cell, 3)),
            (size(cell, 1), size(cell, 3)),
            (size(cell, 1), size(cell, 2)),
            (size(cell, 1), size(cell, 2)),
        )
        cellfaceindices = (
            cellindices[1, 1:end, 1:end],
            cellindices[end, 1:end, 1:end],
            cellindices[1:end, 1, 1:end],
            cellindices[1:end, end, 1:end],
            cellindices[1:end, 1:end, 1],
            cellindices[1:end, 1:end, end],
        )
    else
        error("Unsupported element type $(typeof(cell))")
    end

    numchildfaces = 2^(ndims(cell) - 1)
    faceoffsets = (0, cumsum(prod.(cellfacedims2))...)

    faceorientations = zeros(Raven.Orientation{numchildfaces}, numfaces, numcells)
    for q = 1:numcells
        for (f, faceindices) in enumerate(degree3faceindices)
            tmpface = ntuple(numchildfaces) do n
                dtoc_degree3_global[faceindices[n]..., q]
            end

            faceorientations[f, q] = orient(Val(numchildfaces), tmpface)
        end
    end

    # Note that ghost faces will be connected to themselves
    mapM = reshape(collect(1:(last(faceoffsets)*numcells)), (last(faceoffsets), numcells))
    mapP = reshape(collect(1:(last(faceoffsets)*numcells)), (last(faceoffsets), numcells))

    quadranttoboundary = zeros(Int, numfaces, numcells_local)
    numberofnonconfaces = @MVector zeros(Int, ndims(cell))
    uniquenonconnfaces = @MVector zeros(Int, ndims(cell))
    noncongfacedict = ntuple(ndims(cell)) do _
        Dict{NTuple{numchildfaces,eltype(dtoc_degree3_global)},Int}()
    end
    for q = 1:numcells_local
        for (f, faceindices) in enumerate(degree3faceindices)
            facefirstindex = dtoc_degree3_local[faceindices[1]..., q]
            kind = length(nzrange(ctod_degree3_local, facefirstindex))

            if kind == 1
                quadranttoboundary[f, q] = 1
            elseif kind == 1 + numchildfaces
                # Get the canonical orientation of the global face indices
                fg, _ = fldmod1(f, 2)
                gfaceindices = ntuple(numchildfaces) do n
                    dtoc_degree3_global[faceindices[n]..., q]
                end
                o = faceorientations[f, q]
                gfaceindices = gfaceindices[perm(o)]

                get!(noncongfacedict[fg], gfaceindices) do
                    uniquenonconnfaces[fg] += 1
                end

                numberofnonconfaces[fg] += 1
            end
        end
    end

    # Not sure why I could not use the following.  Type inference
    # was not working.
    #
    # vmapNC = ntuple(Val(ndims(cell))) do n
    #     zeros(
    #         Int,
    #         cellfacedims[n][1:(ndims(cell)-1)]...,
    #         1 + numchildfaces,
    #         uniquenonconnfaces[n],
    #     )
    # end
    #
    # so I ended up just explicitly writing it out

    vmapNC = if ndims(cell) == 2
        (
            zeros(Int, cellfacedims[1][1], 1 + numchildfaces, uniquenonconnfaces[1]),
            zeros(Int, cellfacedims[2][1], 1 + numchildfaces, uniquenonconnfaces[2]),
        )
    else
        (
            zeros(
                Int,
                cellfacedims[1][1],
                cellfacedims[1][2],
                1 + numchildfaces,
                uniquenonconnfaces[1],
            ),
            zeros(
                Int,
                cellfacedims[2][1],
                cellfacedims[2][2],
                1 + numchildfaces,
                uniquenonconnfaces[2],
            ),
            zeros(
                Int,
                cellfacedims[3][1],
                cellfacedims[3][2],
                1 + numchildfaces,
                uniquenonconnfaces[3],
            ),
        )

    end

    ncids = ntuple(ndims(cell)) do n
        zeros(Int, numberofnonconfaces[n])
    end

    nctypes = ntuple(ndims(cell)) do n
        zeros(Int8, numberofnonconfaces[n])
    end

    nctoface = ntuple(ndims(cell)) do n
        zeros(Int8, 2, uniquenonconnfaces[n])
    end

    nonconface = zeros(Int, ndims(cell))
    rows = rowvals(ctod_degree3_local)
    for q = 1:numcells_local
        for (f, faceindices3) in enumerate(degree3faceindices)
            fg = fld1(f, 2)
            facefirstindex = dtoc_degree3_local[faceindices3[1]..., q]
            neighborsrange = nzrange(ctod_degree3_local, facefirstindex)
            kind = length(neighborsrange)

            if kind == 1
                # boundary face

            elseif kind == 2
                # conforming face
                nf = 0
                nq = 0
                for ii in neighborsrange
                    nq, pi = fldmod1(rows[ii], 4^ndims(cell))
                    nf = _indextoface(Val(N), pi)
                    if nq != q || nf != f
                        break
                    end
                end

                # Make sure we found the connecting face
                @assert nf > 0
                @assert nq > 0

                # If faces are paired up from different face groups then
                # make sure that each face has the same number of
                # degrees-of-freedom
                @assert fld1(f, ndims(cell)) == fld1(nf, ndims(cell)) ||
                        length(cellfaceindices[f]) == length(cellfaceindices[nf])

                no = inv(faceorientations[f, q]) ∘ faceorientations[nf, nq]

                for j in CartesianIndices(cellfacedims[fg])
                    nfg = fld1(nf, 2)
                    fij = orientindex(no, cellfacedims[fg], j)
                    mapP[faceoffsets[end]*(q-1)+faceoffsets[f]+facefaceindices[fg][j]] =
                        mapM[faceoffsets[end]*(nq-1)+faceoffsets[nf]+facefaceindices[nfg][fij]]
                end
            else
                # non-conforming face
                @assert kind == 1 + numchildfaces

                nonconface[fg] += 1

                o = faceorientations[f, q]
                gface = ntuple(numchildfaces) do n
                    dtoc_degree3_global[faceindices3[n]..., q]
                end
                gface = gface[perm(o)]
                ncid = noncongfacedict[fg][gface]
                ncids[fg][nonconface[fg]] = ncid

                # compute global quadrant ids that participate in the nonconforming interface
                qs = fld1.(rows[neighborsrange], 4^ndims(cell))
                fs = map(
                    x -> _indextoface(Val(N), x),
                    mod1.(rows[neighborsrange], 4^ndims(cell)),
                )

                # use level to figure out which ones are the smaller ones
                ls = quadranttolevel[qs]
                childlevel = maximum(ls)

                cids = findall(==(childlevel), ls)
                pids = findall(==(childlevel - 1), ls)

                @assert length(cids) == numchildfaces
                @assert length(pids) == 1

                childquadrants = qs[cids]
                childquadrants =
                    childquadrants[sortperm(quadranttoglobalid[childquadrants])]
                parentquadrant = first(qs[pids])

                # use orientation to transform the elements order
                childface = fs[first(cids)]
                childorientation = faceorientations[childface, first(childquadrants)]
                childquadrants = childquadrants[perm(childorientation)]
                nctype = if q == parentquadrant
                    1
                else
                    something(findfirst(==(q), childquadrants), 0) + 1
                end
                nctypes[fg][nonconface[fg]] = nctype

                if vmapNC[fg][ntuple((_->1), ndims(cell)-1)..., 1, ncid] == 0
                    # fill the non-conforming group
                    parentface = fs[first(pids)]
                    parentorientation = faceorientations[parentface, parentquadrant]

                    nctoface[fg][1, ncid] = parentface
                    nctoface[fg][2, ncid] = childface
                    for j in CartesianIndices(cellfacedims[fg])
                        pfi = orientindex(parentorientation, cellfacedims[fg], j)
                        cfi = orientindex(childorientation, cellfacedims[fg], j)
                        vmapNC[fg][j, 1, ncid] =
                            cellfaceindices[parentface][pfi] +
                            (parentquadrant - 1) * length(cell)
                        for c = 1:numchildfaces
                            vmapNC[fg][j, c+1, ncid] =
                                cellfaceindices[childface][cfi] +
                                (childquadrants[c] - 1) * length(cell)
                        end
                    end
                end
            end
        end
    end

    vmapM = zeros(Int, size(mapM))
    for q = 1:numcells,
        fg = 1:ndims(cell),
        fn = 1:2,
        j in CartesianIndices(cellfacedims[fg])

        f = 2 * (fg - 1) + fn
        idx = faceoffsets[end] * (q - 1) + faceoffsets[f] + facefaceindices[fg][j]
        vmapM[idx] = cellfaceindices[f][j] + (q - 1) * length(cell)
    end
    vmapP = zeros(Int, size(mapM))
    for n in eachindex(vmapP, vmapM, mapP)
        vmapP[n] = vmapM[mapP[n]]
    end

    return (; vmapM, vmapP, mapM, mapP, vmapNC, nctoface, nctypes, ncids),
    quadranttoboundary
end

function materializenodecommpattern(cell::LobattoCell, ctod, quadrantcommpattern)
    ghostranktompirank = quadrantcommpattern.recvranks
    ghostranktoindices = expand.(
        [
            quadrantcommpattern.recvindices[ids] for
            ids in quadrantcommpattern.recvrankindices
        ],
        length(cell),
    )

    ranktype = eltype(ghostranktompirank)
    indicestype = eltype(eltype(ghostranktoindices))

    if length(ghostranktompirank) == 0
        return CommPattern{Array}(
            indicestype[],
            ranktype[],
            UnitRange{indicestype}[],
            indicestype[],
            ranktype[],
            UnitRange{indicestype}[],
        )

    end

    dofstarts = zeros(indicestype, length(ghostranktoindices) + 1)
    for (i, ids) in enumerate(ghostranktoindices)
        dofstarts[i] = first(ids)
    end
    dofstarts[end] = last(last(ghostranktoindices)) + 0x1

    senddofs = Dict{ranktype,Set{indicestype}}()
    recvdofs = Dict{ranktype,Set{indicestype}}()

    rows = rowvals(ctod)
    n = size(ctod, 2)
    remoteranks = Set{ranktype}()
    for j = 1:n
        containslocal = false
        for k in nzrange(ctod, j)
            i = rows[k]
            s = searchsorted(dofstarts, i)
            if last(s) > 0
                push!(remoteranks, ghostranktompirank[last(s)])
            end
            if last(s) == 0
                containslocal = true
            end
        end

        if !isempty(remoteranks)
            for k in nzrange(ctod, j)
                i = rows[k]
                s = searchsorted(dofstarts, i)
                if last(s) == 0
                    for rank in remoteranks
                        # local node we need to send
                        sendset = get!(senddofs, rank) do
                            Set{indicestype}()
                        end
                        push!(sendset, i)
                    end
                elseif containslocal
                    # remote node we need to recv
                    rank = ghostranktompirank[last(s)]
                    recvset = get!(recvdofs, rank) do
                        Set{indicestype}()
                    end
                    push!(recvset, i)
                end
            end
        end

        empty!(remoteranks)
    end

    numsendindices = 0
    for dofs in keys(senddofs)
        numsendindices += length(dofs)
    end

    sendindices = Int[]
    sendrankindices = UnitRange{Int}[]
    sendoffset = 0
    for r in ghostranktompirank
        dofs = senddofs[r]
        append!(sendindices, sort(collect(dofs)))
        push!(sendrankindices, (1:length(dofs)) .+ sendoffset)

        sendoffset += length(dofs)
    end

    recvindices = Int[]
    recvrankindices = UnitRange{Int}[]
    recvoffset = 0
    for r in ghostranktompirank
        dofs = recvdofs[r]
        append!(recvindices, sort(collect(dofs)))
        push!(recvrankindices, (1:length(dofs)) .+ recvoffset)

        recvoffset += length(dofs)
    end

    return CommPattern{Array}(
        recvindices,
        ghostranktompirank,
        recvrankindices,
        sendindices,
        ghostranktompirank,
        sendrankindices,
    )
end

function materializeparentnodes(
    cell::LobattoCell,
    ctod,
    quadranttoglobalid,
    quadranttolevel,
)
    Np = length(cell)
    rows = rowvals(ctod)
    m, n = size(ctod)
    parentdofs = zeros(eltype(rows), m)
    for j = 1:n
        level = typemax(Int8)
        gid = typemax(eltype(quadranttoglobalid))
        pdof = 0
        for ii in nzrange(ctod, j)
            i = rows[ii]
            e = cld(i, Np)
            if quadranttolevel[e] ≤ level && quadranttoglobalid[e] < gid
                level = quadranttolevel[e]
                gid = quadranttoglobalid[e]
                pdof = i
            end
        end

        for ii in nzrange(ctod, j)
            i = rows[ii]
            @assert pdof != 0
            parentdofs[i] = pdof
        end
    end

    return reshape(parentdofs, size(cell)..., :)
end

@kernel function linevolumemetrics!(
    firstordermetrics,
    secondordermetrics,
    points,
    Dr,
    wr,
    ::Val{IR},
    ::Val{Q},
) where {IR,Q}
    i, p = @index(Local, NTuple)
    _, q = @index(Global, NTuple)

    @uniform T = eltype(points)

    X = @localmem T (IR, Q)
    dXdr = @private T (1,)
    dXds = @private T (1,)

    @inbounds begin
        X[i, p] = points[i, q]

        @synchronize

        dXdr[] = -zero(T)

        @unroll for m = 1:IR
            dXdr[] += Dr[i, m] * X[m, p]
        end

        G = dXdr[] * dXdr[]
        invG = inv(G)

        dRdX = invG * dXdr[]

        J = norm(dXdr[])

        wJ = wr[i] * J

        wJinvG = wJ * invG

        firstordermetrics[i, j, q] = (; dRdX, J, wJ)
        secondordermetrics[i, j, q] = (; wJinvG, wJ)
    end
end

@kernel function linevolumebrickmetrics!(
    firstordermetrics,
    secondordermetrics,
    points,
    wr,
    ::Val{IR},
    ::Val{Q},
) where {IR,Q}
    i, p = @index(Local, NTuple)
    _, q = @index(Global, NTuple)

    @uniform T = eltype(points)

    X = @localmem T (IR, Q)

    @inbounds begin
        X[i, p] = points[i, q]

        @synchronize

        dx = X[end, p][1] - X[1, p][1]
        dr = 2

        drdx = dr / dx

        dRdX = drdx

        J = (dx / dr)

        wJ = wr[i] * J

        # invwJ = inv(wJ)

        wJinvG = wJ * (drdx)^2

        firstordermetrics[i, q] = (; dRdX, J, wJ)
        secondordermetrics[i, q] = (; wJinvG, wJ)
    end
end

@kernel function linesurfacemetrics!(
    surfacemetrics,
    dRdXs,
    Js,
    _,
    vmapM,
    w,
    fg,
    facegroupsize,
    facegroupoffsets,
)
    j, fn, _ = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @inbounds begin
        si =
            facegroupoffsets[end] * (q - 1) +
            facegroupoffsets[fg] +
            (fn - 1) * facegroupsize[1] +
            j
        i = vmapM[si]
        dRdX = dRdXs[i]
        J = Js[i]

        sn = fn == 1 ? -1 : 1

        n = sn * J * dRdX
        sJ = norm(n)
        n = SA[n/sJ]

        wsJ = sJ # weight is 1 for faces with one point

        surfacemetrics[si] = (; n, sJ, wsJ)
    end
end

function materializemetrics(
    cell::LobattoLine,
    points::AbstractArray{SVector{N,FT}},
    facemaps,
    comm,
    nodecommpattern,
    isunwarpedbrick,
) where {N,FT}
    Q = max(512 ÷ prod(size(cell)), 1)

    D = derivatives_1d(cell)
    w = vec.(weights_1d(cell))

    T1 = NamedTuple{(:dRdX, :J, :wJ),Tuple{FT,FT,FT}}
    firstordermetrics = similar(points, T1)

    T2 = NamedTuple{(:wJinvG, :wJ),Tuple{FT,FT}}
    secondordermetrics = similar(points, T2)

    backend = get_backend(points)

    if isunwarpedbrick
        kernel! = linevolumebrickmetrics!(backend, (size(cell)..., Q))
        kernel!(
            firstordermetrics,
            secondordermetrics,
            points,
            w...,
            Val.(size(cell))...,
            Val(Q);
            ndrange = size(points),
        )
    else
        kernel! = linevolumemetrics!(backend, (size(cell)..., Q))
        kernel!(
            firstordermetrics,
            secondordermetrics,
            points,
            D...,
            w...,
            Val.(size(cell))...,
            Val(Q);
            ndrange = size(points),
        )
    end

    fcm = commmanager(eltype(firstordermetrics), nodecommpattern; comm)
    share!(firstordermetrics, fcm)

    volumemetrics = (firstordermetrics, secondordermetrics)

    TS = NamedTuple{(:n, :sJ, :wsJ),Tuple{SVector{N,FT},FT,FT}}

    # TODO move this to cell
    cellfacedims = (1,)
    facegroupsize = cellfacedims
    facegroupoffsets = (0, cumsum(2 .* prod.(cellfacedims))...)

    surfacemetrics = GridArray{TS}(
        undef,
        arraytype(points),
        (facegroupoffsets[end], sizewithoutghosts(points)[end]),
        (facegroupoffsets[end], sizewithghosts(points)[end]),
        Raven.comm(points),
        true,
        1,
    )

    for n in eachindex(cellfacedims)
        J = cellfacedims[n]
        Q = max(512 ÷ 2prod(J), 1)

        kernel! = linesurfacemetrics!(backend, (J..., 2, Q))
        kernel!(
            surfacemetrics,
            components(viewwithghosts(firstordermetrics))...,
            facemaps.vmapM,
            w,
            n,
            facegroupsize[n],
            facegroupoffsets,
            ndrange = (J..., 2, last(size(surfacemetrics))),
        )
    end

    surfacemetrics = (viewwithoutghosts(surfacemetrics), nothing)

    return (volumemetrics, surfacemetrics)
end

@kernel function quadvolumemetrics!(
    firstordermetrics,
    secondordermetrics,
    points,
    Dr,
    Ds,
    wr,
    ws,
    ::Val{IR},
    ::Val{IS},
    ::Val{Q},
) where {IR,IS,Q}
    i, j, p = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(points)

    X = @localmem T (IR, IS, Q)
    dXdr = @private T (1,)
    dXds = @private T (1,)

    @inbounds begin
        X[i, j, p] = points[i, j, q]

        @synchronize

        dXdr[] = -zero(T)
        dXds[] = -zero(T)

        @unroll for m = 1:IR
            dXdr[] += Dr[i, m] * X[m, j, p]
        end

        @unroll for n = 1:IS
            dXds[] += Ds[j, n] * X[i, n, p]
        end

        G = SMatrix{2,2,eltype(Dr),4}(
            dot(dXdr[], dXdr[]),
            dot(dXdr[], dXds[]),
            dot(dXdr[], dXds[]),
            dot(dXds[], dXds[]),
        )

        invG = inv(G)

        dRdX = invG * [dXdr[] dXds[]]'

        J = norm(cross(dXdr[], dXds[]))

        wJ = wr[i] * ws[j] * J

        wJinvG = wJ * invG

        firstordermetrics[i, j, q] = (; dRdX, J, wJ)
        secondordermetrics[i, j, q] = (; wJinvG, wJ)
    end
end

@kernel function quadvolumebrickmetrics!(
    firstordermetrics,
    secondordermetrics,
    points,
    wr,
    ws,
    ::Val{IR},
    ::Val{IS},
    ::Val{Q},
) where {IR,IS,Q}
    i, j, p = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(points)

    X = @localmem T (IR, IS, Q)

    @inbounds begin
        X[i, j, p] = points[i, j, q]

        @synchronize

        dx = X[end, 1, p][1] - X[1, 1, p][1]
        dy = X[1, end, p][2] - X[1, 1, p][2]
        dr = 2
        ds = 2

        drdx = dr / dx
        dsdy = ds / dy

        dRdX = SMatrix{2,2,typeof(dx),4}(drdx, -zero(dx), -zero(dx), dsdy)

        J = (dx / dr) * (dy / ds)

        wJ = wr[i] * ws[j] * J

        # invwJ = inv(wJ)

        wJinvG = SMatrix{2,2,typeof(dx),4}(
            SVector(wJ * (drdx)^2, -zero(dx), -zero(dx), wJ * (dsdy)^2),
        )

        firstordermetrics[i, j, q] = (; dRdX, J, wJ)
        secondordermetrics[i, j, q] = (; wJinvG, wJ)
    end
end

@kernel function quadsurfacemetrics!(
    surfacemetrics,
    dRdXs,
    Js,
    _,
    vmapM,
    w,
    fg,
    facegroupsize,
    facegroupoffsets,
)
    j, fn, _ = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @inbounds begin
        si =
            facegroupoffsets[end] * (q - 1) +
            facegroupoffsets[fg] +
            (fn - 1) * facegroupsize[1] +
            j
        i = vmapM[si]
        dRdX = dRdXs[i]
        J = Js[i]

        sn = fn == 1 ? -1 : 1

        n = sn * J * dRdX[fg, :]
        sJ = norm(n)
        n = n / sJ

        if fg == 1
            wsJ = sJ * w[2][j]
        elseif fg == 2
            wsJ = sJ * w[1][j]
        end

        surfacemetrics[si] = (; n, sJ, wsJ)
    end
end

@kernel function quadncsurfacemetrics!(surfacemetrics, dRdXs, Js, _, vmapNC, nctoface, w)
    j, ft, _ = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @inbounds begin
        i = vmapNC[j, ft, q]
        dRdX = dRdXs[i]
        J = Js[i]

        f = ft == 1 ? nctoface[1, q] : nctoface[2, q]
        fg, fn = fldmod1(f, 2)

        sn = fn == 1 ? -1 : 1

        n = sn * J * dRdX[fg, :]
        sJ = norm(n)
        n = n / sJ

        if fg == 1
            wsJ = sJ * w[2][j]
        elseif fg == 2
            wsJ = sJ * w[1][j]
        end

        surfacemetrics[j, ft, q] = (; n, sJ, wsJ)
    end
end


function materializemetrics(
    cell::LobattoQuad,
    points::AbstractArray{SVector{N,FT}},
    facemaps,
    comm,
    nodecommpattern,
    isunwarpedbrick,
) where {N,FT}
    Q = max(512 ÷ prod(size(cell)), 1)

    D = derivatives_1d(cell)
    w = vec.(weights_1d(cell))

    T1 = NamedTuple{(:dRdX, :J, :wJ),Tuple{SMatrix{2,N,FT,2 * N},FT,FT}}
    firstordermetrics = similar(points, T1)

    T2 = NamedTuple{(:wJinvG, :wJ),Tuple{SHermitianCompact{2,FT,3},FT}}
    secondordermetrics = similar(points, T2)

    backend = get_backend(points)

    if isunwarpedbrick
        kernel! = quadvolumebrickmetrics!(backend, (size(cell)..., Q))
        kernel!(
            firstordermetrics,
            secondordermetrics,
            points,
            w...,
            Val.(size(cell))...,
            Val(Q);
            ndrange = size(points),
        )
    else
        kernel! = quadvolumemetrics!(backend, (size(cell)..., Q))
        kernel!(
            firstordermetrics,
            secondordermetrics,
            points,
            D...,
            w...,
            Val.(size(cell))...,
            Val(Q);
            ndrange = size(points),
        )
    end

    fcm = commmanager(eltype(firstordermetrics), nodecommpattern; comm)
    share!(firstordermetrics, fcm)

    volumemetrics = (firstordermetrics, secondordermetrics)

    TS = NamedTuple{(:n, :sJ, :wsJ),Tuple{SVector{N,FT},FT,FT}}

    # TODO move this to cell
    cellfacedims = ((size(cell, 2),), (size(cell, 1),))
    facegroupsize = cellfacedims
    facegroupoffsets = (0, cumsum(2 .* prod.(cellfacedims))...)

    surfacemetrics = GridArray{TS}(
        undef,
        arraytype(points),
        (facegroupoffsets[end], sizewithoutghosts(points)[end]),
        (facegroupoffsets[end], sizewithghosts(points)[end]),
        Raven.comm(points),
        true,
        1,
    )

    ncsurfacemetrics = ntuple(
        n -> similar(points, TS, size(facemaps.vmapNC[n])),
        Val(length(cellfacedims)),
    )

    for n in eachindex(cellfacedims)
        J = cellfacedims[n]
        Q = max(512 ÷ 2prod(J), 1)

        kernel! = quadsurfacemetrics!(backend, (J..., 2, Q))
        kernel!(
            surfacemetrics,
            components(viewwithghosts(firstordermetrics))...,
            facemaps.vmapM,
            w,
            n,
            facegroupsize[n],
            facegroupoffsets,
            ndrange = (J..., 2, last(size(surfacemetrics))),
        )

        M = 1 + 2^(ndims(cell) - 1)
        Q = max(512 ÷ (M * prod(J)), 1)
        kernel! = quadncsurfacemetrics!(backend, (J..., M, Q))
        kernel!(
            ncsurfacemetrics[n],
            components(viewwithghosts(firstordermetrics))...,
            facemaps.vmapNC[n],
            facemaps.nctoface[n],
            w,
            ndrange = size(ncsurfacemetrics[n]),
        )
    end

    surfacemetrics = (viewwithoutghosts(surfacemetrics), ncsurfacemetrics)

    return (volumemetrics, surfacemetrics)
end

@kernel function hexvolumemetrics!(
    firstordermetrics,
    secondordermetrics,
    points,
    Dr,
    Ds,
    Dt,
    wr,
    ws,
    wt,
    ::Val{IR},
    ::Val{IS},
    ::Val{IT},
    ::Val{Q},
) where {IR,IS,IT,Q}
    i, j, p = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(points)
    @uniform FT = eltype(eltype(points))

    vtmp = @localmem T (IR, IS, Q)

    X = @private T (IT,)
    dXdr = @private T (IT,)
    dXds = @private T (IT,)
    dXdt = @private T (IT,)

    a1 = @private T (1,)
    a2 = @private T (1,)
    a3 = @private T (1,)

    @inbounds begin
        @unroll for k = 1:IT
            dXdr[k] = -zero(T)
            dXds[k] = -zero(T)
            dXdt[k] = -zero(T)
        end

        @unroll for k = 1:IT
            X[k] = points[i, j, k, q]

            vtmp[i, j, p] = X[k]

            @synchronize

            @unroll for m = 1:IR
                dXdr[k] += Dr[i, m] * vtmp[m, j, p]
            end

            @unroll for n = 1:IS
                dXds[k] += Ds[j, n] * vtmp[i, n, p]
            end

            @unroll for o = 1:IT
                dXdt[o] += Dt[o, k] * vtmp[i, j, p]
            end

            @synchronize
        end

        @unroll for k = 1:IT

            # Instead of
            # ```julia
            # @. invJ = inv(J)
            # ```
            # we use the curl invariant formulation of Kopriva, equation (37) of
            # <https://doi.org/10.1007/s10915-005-9070-8>.

            a1[] = -zero(T) # Ds * cross(X, dXdt) - Dt * cross(X, dXds)
            a2[] = -zero(T) # Dt * cross(X, dXdr) - Dr * cross(X, dXdt)
            a3[] = -zero(T) # Dr * cross(X, dXds) - Ds * cross(X, dXdr)

            # Dr * cross(X, dXds)
            @synchronize

            vtmp[i, j, p] = cross(X[k], dXds[k])

            @synchronize

            @unroll for m = 1:IR
                a3[] += Dr[i, m] * vtmp[m, j, p]
            end

            # Dr * cross(X, dXdt)
            @synchronize

            vtmp[i, j, p] = cross(X[k], dXdt[k])

            @synchronize

            @unroll for m = 1:IR
                a2[] -= Dr[i, m] * vtmp[m, j, p]
            end


            # Ds * cross(X, dXdt)
            @synchronize

            vtmp[i, j, p] = cross(X[k], dXdt[k])

            @synchronize

            @unroll for n = 1:IS
                a1[] += Ds[j, n] * vtmp[i, n, p]
            end

            # Ds * cross(X, dXdr)
            @synchronize

            vtmp[i, j, p] = cross(X[k], dXdr[k])

            @synchronize

            @unroll for n = 1:IS
                a3[] -= Ds[j, n] * vtmp[i, n, p]
            end

            # Dt * cross(X, dXdr)
            @unroll for o = 1:IT
                a2[] += Dt[k, o] * cross(X[o], dXdr[o])
            end

            # Dt * cross(X, dXds)
            @unroll for o = 1:IT
                a1[] -= Dt[k, o] * cross(X[o], dXds[o])
            end

            J = [dXdr[k] dXds[k] dXdt[k]]
            detJ = det(J)
            invJ = [a1[] a2[] a3[]]' ./ 2detJ


            invG = invJ * invJ'
            wJ = wr[i] * ws[j] * wt[k] * detJ
            wJinvG = wJ * invG

            firstordermetrics[i, j, k, q] = (; dRdX = invJ, J = detJ, wJ)
            secondordermetrics[i, j, k, q] = (; wJinvG, wJ)
        end
    end
end

@kernel function hexvolumebrickmetrics!(
    firstordermetrics,
    secondordermetrics,
    points,
    wr,
    ws,
    wt,
    ::Val{IR},
    ::Val{IS},
    ::Val{IT},
    ::Val{Q},
) where {IR,IS,IT,Q}
    i, j, p = @index(Local, NTuple)
    _, _, q = @index(Global, NTuple)

    @uniform T = eltype(points)
    @uniform FT = eltype(eltype(points))


    @inbounds begin
        x = points[1, 1, 1, p]

        dx = points[end, 1, 1, p][1] - x[1]
        dy = points[1, end, 1, p][2] - x[2]
        dz = points[1, 1, end, p][3] - x[3]

        dr = 2
        ds = 2
        dt = 2

        drdx = dr / dx
        dsdy = ds / dy
        dtdz = dt / dz

        dRdX = SMatrix{3,3,typeof(dx),9}(
            drdx,
            -zero(dx),
            -zero(dx),
            -zero(dx),
            dsdy,
            -zero(dx),
            -zero(dx),
            -zero(dx),
            dtdz,
        )

        J = (dx / dr) * (dy / ds) * (dz / dt)

        @unroll for k = 1:IT
            wJ = wr[i] * ws[j] * wt[k] * J

            # invwJ = inv(wJ)

            wJinvG = SMatrix{3,3,typeof(dx),9}(
                wJ * (drdx)^2,
                -zero(dx),
                -zero(dx),
                -zero(dx),
                wJ * (dsdy)^2,
                -zero(dx),
                -zero(dx),
                -zero(dx),
                wJ * (dtdz)^2,
            )

            firstordermetrics[i, j, k, q] = (; dRdX, J, wJ)
            secondordermetrics[i, j, k, q] = (; wJinvG, wJ)
        end
    end
end

@kernel function hexsurfacemetrics!(
    surfacemetrics,
    dRdXs,
    Js,
    _,
    vmapM,
    w,
    fg,
    facegroupsize,
    facegroupoffsets,
)
    i, j, fn, _ = @index(Local, NTuple)
    _, _, _, q = @index(Global, NTuple)

    begin
        sk =
            facegroupoffsets[end] * (q - 1) +
            facegroupoffsets[fg] +
            (fn - 1) * facegroupsize[1] * facegroupsize[2] +
            (j - 1) * facegroupsize[1] +
            i
        k = vmapM[sk]
        dRdX = dRdXs[k]
        J = Js[k]

        sn = fn == 1 ? -1 : 1

        n = sn * J * dRdX[fg, :]

        sJ = norm(n)
        n = n / sJ

        if fg == 1
            wsJ = sJ * w[2][i] * w[3][j]
        elseif fg == 2
            wsJ = sJ * w[1][i] * w[3][j]
        elseif fg == 3
            wsJ = sJ * w[1][i] * w[2][j]
        end

        surfacemetrics[sk] = (; n, sJ, wsJ)
    end
end

@kernel function hexncsurfacemetrics!(surfacemetrics, dRdXs, Js, _, vmapNC, nctoface, w)
    i, j, ft, _ = @index(Local, NTuple)
    _, _, _, q = @index(Global, NTuple)

    @inbounds begin
        k = vmapNC[i, j, ft, q]
        dRdX = dRdXs[k]
        J = Js[k]

        f = ft == 1 ? nctoface[1, q] : nctoface[2, q]
        fg, fn = fldmod1(f, 2)

        sn = fn == 1 ? -1 : 1

        n = sn * J * dRdX[fg, :]
        sJ = norm(n)
        n = n / sJ

        if fg == 1
            wsJ = sJ * w[2][i] * w[3][j]
        elseif fg == 2
            wsJ = sJ * w[1][i] * w[3][j]
        elseif fg == 3
            wsJ = sJ * w[1][i] * w[2][j]
        end

        surfacemetrics[i, j, ft, q] = (; n, sJ, wsJ)
    end
end

function materializemetrics(
    cell::LobattoHex,
    points::AbstractArray{SVector{3,FT}},
    facemaps,
    comm,
    nodecommpattern,
    isunwarpedbrick,
) where {FT}
    Q = max(512 ÷ (size(cell, 1) * size(cell, 2)), 1)

    D = derivatives_1d(cell)
    w = vec.(weights_1d(cell))

    T1 = NamedTuple{(:dRdX, :J, :wJ),Tuple{SMatrix{3,3,FT,9},FT,FT}}
    firstordermetrics = similar(points, T1)

    T2 = NamedTuple{(:wJinvG, :wJ),Tuple{SHermitianCompact{3,FT,6},FT}}
    secondordermetrics = similar(points, T2)

    backend = get_backend(points)

    if isunwarpedbrick
        kernel! = hexvolumebrickmetrics!(backend, (size(cell, 1), size(cell, 2), Q))
        kernel!(
            firstordermetrics,
            secondordermetrics,
            points,
            w...,
            Val.(size(cell))...,
            Val(Q);
            ndrange = (size(cell, 1), size(cell, 2), size(points, 4)),
        )
    else
        kernel! = hexvolumemetrics!(backend, (size(cell, 1), size(cell, 2), Q))
        kernel!(
            firstordermetrics,
            secondordermetrics,
            points,
            D...,
            w...,
            Val.(size(cell))...,
            Val(Q);
            ndrange = (size(cell, 1), size(cell, 2), size(points, 4)),
        )
    end

    # We need this to compute the normals and surface Jacobian on the
    # non-conforming faces.  Should we change the nonconforming surface
    # information to avoid this communication?
    fcm = commmanager(eltype(firstordermetrics), nodecommpattern; comm)
    share!(firstordermetrics, fcm)

    volumemetrics = (firstordermetrics, secondordermetrics)

    TS = NamedTuple{(:n, :sJ, :wsJ),Tuple{SVector{3,FT},FT,FT}}

    # TODO move this to cell
    cellfacedims = (
        (size(cell, 2), size(cell, 3)),
        (size(cell, 1), size(cell, 3)),
        (size(cell, 1), size(cell, 2)),
    )
    facegroupsize = cellfacedims
    facegroupoffsets = (0, cumsum(2 .* prod.(cellfacedims))...)

    surfacemetrics = GridArray{TS}(
        undef,
        arraytype(points),
        (facegroupoffsets[end], sizewithoutghosts(points)[end]),
        (facegroupoffsets[end], sizewithghosts(points)[end]),
        Raven.comm(points),
        false,
        1,
    )

    ncsurfacemetrics = ntuple(
        n -> similar(points, TS, size(facemaps.vmapNC[n])),
        Val(length(cellfacedims)),
    )

    for n in eachindex(cellfacedims)
        J = cellfacedims[n]
        Q = max(512 ÷ prod(J), 1)

        kernel! = hexsurfacemetrics!(backend, (J..., 2, Q))
        kernel!(
            surfacemetrics,
            components(viewwithghosts(firstordermetrics))...,
            facemaps.vmapM,
            w,
            n,
            facegroupsize[n],
            facegroupoffsets,
            ndrange = (J..., 2, last(size(surfacemetrics))),
        )

        M = 1 + 2^(ndims(cell) - 1)
        Q = max(512 ÷ (M * prod(J)), 1)
        kernel! = hexncsurfacemetrics!(backend, (J..., M, Q))
        kernel!(
            ncsurfacemetrics[n],
            components(viewwithghosts(firstordermetrics))...,
            facemaps.vmapNC[n],
            facemaps.nctoface[n],
            w,
            ndrange = size(ncsurfacemetrics[n]),
        )
    end

    surfacemetrics = (viewwithoutghosts(surfacemetrics), ncsurfacemetrics)

    return (volumemetrics, surfacemetrics)
end
