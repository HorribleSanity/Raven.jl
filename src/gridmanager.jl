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

struct GridManager{C<:AbstractCell,G<:AbstractCoarseGrid,V,P} <: AbstractGridManager
    referencecell::C
    coarsegrid::G
    coarsegridvertices::V
    forest::P
end

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

    coarsegridvertices = adapt(arraytype(referencecell), vertices(coarsegrid))

    return GridManager(referencecell, coarsegrid, coarsegridvertices, p)
end

Base.length(gm::GridManager) = P4estTypes.lengthoflocalquadrants(gm.forest)

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

    P4estTypes.iterateforest(gm.forest; userdata = flags, volume = fill_quadrant_user_data)

    P4estTypes.coarsen!(gm.forest; coarsen = coarsen_quads, replace = replace_quads)
    P4estTypes.refine!(gm.forest; refine = refine_quads, replace = replace_quads)
    P4estTypes.balance!(gm.forest; replace = replace_quads)

    P4estTypes.iterateforest(gm.forest; userdata = flags, volume = fill_adapt_flags)

    return
end
