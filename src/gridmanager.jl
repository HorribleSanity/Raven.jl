abstract type AbstractGridManager end

struct QuadData
    old_id::P4estTypes.Locidx
    old_level::Int8
    adapt_flag::UInt8
end

const AdaptCoarsen = 0x00
const AdaptNone = 0x01
const AdaptRefine = 0x10
const AdaptTouched = 0x11

struct GridManager{C<:AbstractCell,G<:AbstractCoarseGrid,E,V,P} <: AbstractGridManager
    comm::MPI.Comm
    referencecell::C
    coarsegrid::G
    coarsegridcells::E
    coarsegridvertices::V
    forest::P
end

comm(gm::GridManager) = gm.comm
referencecell(gm::GridManager) = gm.referencecell
coarsegrid(gm::GridManager) = gm.coarsegrid
coarsegridcells(gm::GridManager) = gm.coarsegridcells
coarsegridvertices(gm::GridManager) = gm.coarsegridvertices
forest(gm::GridManager) = gm.forest

function GridManager(
    referencecell,
    coarsegrid;
    comm = MPI.COMM_WORLD,
    min_level = 0,
    fill_uniform = true,
)
    p = P4estTypes.pxest(
        connectivity(coarsegrid);
        min_level = first(min_level),
        data_type = QuadData,
        comm,
        fill_uniform,
    )

    coarsegridcells = adapt(arraytype(referencecell), cells(coarsegrid))
    coarsegridvertices = adapt(arraytype(referencecell), vertices(coarsegrid))

    return GridManager(
        comm,
        referencecell,
        coarsegrid,
        coarsegridcells,
        coarsegridvertices,
        p,
    )
end



Base.length(gm::GridManager) = P4estTypes.lengthoflocalquadrants(forest(gm))

function fill_quadrant_user_data(forest, _, quadrant, quadrantid, treeid, flags)
    id = quadrantid + P4estTypes.offset(forest[treeid])
    P4estTypes.storeuserdata!(quadrant, QuadData(id, P4estTypes.level(quadrant), flags[id]))
end

function fill_adapt_flags(::P4estTypes.Pxest{X}, _, quadrant, _, _, flags) where {X}
    d = P4estTypes.loaduserdata(quadrant)

    if d.old_level < P4estTypes.level(quadrant)
        flags[d.old_id] = AdaptRefine
    elseif d.old_level > P4estTypes.level(quadrant)
        flags[d.old_id] = AdaptCoarsen
    else
        flags[d.old_id] = AdaptNone
    end

    return
end

function replace_quads(_, _, outgoing, incoming)
    outd = P4estTypes.loaduserdata(first(outgoing))

    for quadrant in incoming
        P4estTypes.storeuserdata!(
            quadrant,
            QuadData(outd.old_id, outd.old_level, AdaptTouched),
        )
    end

    return
end

function refine_quads(_, _, quadrant)
    retval = P4estTypes.loaduserdata(quadrant).adapt_flag == AdaptRefine
    return retval
end

function coarsen_quads(_, _, children)
    coarsen = true

    for child in children
        coarsen &= P4estTypes.loaduserdata(child).adapt_flag == AdaptCoarsen
    end

    return coarsen
end

function adapt!(gm::GridManager, flags)
    @assert length(gm) == length(flags)

    P4estTypes.iterateforest(forest(gm); userdata = flags, volume = fill_quadrant_user_data)

    P4estTypes.coarsen!(forest(gm); coarsen = coarsen_quads, replace = replace_quads)
    P4estTypes.refine!(forest(gm); refine = refine_quads, replace = replace_quads)
    P4estTypes.balance!(forest(gm); replace = replace_quads)

    P4estTypes.iterateforest(forest(gm); userdata = flags, volume = fill_adapt_flags)

    return
end

generate(gm::GridManager) = generate(identity, gm)

function materializeforestvolumedata(forest, _, quadrant, quadid, treeid, data)
    id = quadid + P4estTypes.offset(forest[treeid])

    data.quadranttolevel[id] = P4estTypes.level(quadrant)
    data.quadranttotreeid[id] = treeid
    data.quadranttocoordinate[id, :] .= P4estTypes.coordinates(quadrant)

    return
end

function generate(warp::Function, gm::GridManager)
    # Need to get integer coordinates of cells

    A = arraytype(referencecell(gm))

    quadranttolevel = Array{Int8}(undef, length(gm))
    quadranttotreeid = Array{Int32}(undef, length(gm))
    quadranttocoordinate =
        Array{Int32}(undef, length(gm), P4estTypes.quadrantndims(forest(gm)))
    data = (; quadranttolevel, quadranttotreeid, quadranttocoordinate)

    pin.(A, values(data))

    P4estTypes.iterateforest(
        forest(gm);
        userdata = data,
        volume = materializeforestvolumedata,
    )

    # Send data to the device
    quadranttolevel = A(quadranttolevel)
    quadranttotreeid = A(quadranttotreeid)
    quadranttocoordinate = A(quadranttocoordinate)

    points = materializepoints(
        referencecell(gm),
        coarsegridcells(gm),
        coarsegridvertices(gm),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
    )

    points = warp.(points)

    part = MPI.Comm_rank(comm(gm)) + 1
    nparts = MPI.Comm_size(comm(gm))
    offset = MPI.Scan(Int(length(gm)), MPI.MPI_SUM, MPI.COMM_WORLD) - length(gm)

    return Grid(
        part,
        nparts,
        referencecell(gm),
        offset,
        points,
        quadranttolevel,
        quadranttotreeid,
    )
end
