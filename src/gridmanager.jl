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
    if !MPI.Initialized()
        threadlevel = usetriplebuffer(arraytype(referencecell)) ? :multiple : :serialized
        MPI.Init(; threadlevel)
    end

    comm = MPI.Comm_dup(comm)

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

function _offsets_to_ranges(offsets; by = identity)
    indices = Cint[]
    ranges = UnitRange{Int}[]
    for r = 1:length(offsets)-1
        if offsets[r+1] - offsets[r] > 0
            ids = by(r - 1)
            push!(indices, ids)
            push!(ranges, (offsets[r]+1):offsets[r+1])
        end
    end

    return (indices, ranges)
end

function materializequadrantcommpattern(forest, ghost)
    ms = P4estTypes.mirrors(ghost)
    gs = P4estTypes.ghosts(ghost)

    GC.@preserve ghost begin
        mirror_proc_mirrors = P4estTypes.unsafe_mirror_proc_mirrors(ghost)
        mirror_proc_offsets = P4estTypes.unsafe_mirror_proc_offsets(ghost)

        num_local = P4estTypes.lengthoflocalquadrants(forest)
        T = typeof(num_local)
        num_ghosts = T(length(gs))

        sendindices = similar(mirror_proc_mirrors, T)
        for i in eachindex(sendindices)
            sendindices[i] = P4estTypes.unsafe_local_num(ms[mirror_proc_mirrors[i]+1])
        end
        (sendranks, sendrankindices) = _offsets_to_ranges(mirror_proc_offsets)

        proc_offsets = P4estTypes.unsafe_proc_offsets(ghost)

        recvindices = (num_local+0x1):(num_local+num_ghosts)
        (recvranks, recvrankindices) = _offsets_to_ranges(proc_offsets)
    end

    return CommPattern{Array}(
        recvindices,
        recvranks,
        recvrankindices,
        sendindices,
        sendranks,
        sendrankindices,
    )
end

function materializeforestvolumedata(forest, _, quadrant, quadid, treeid, data)
    id = quadid + P4estTypes.offset(forest[treeid])

    data.quadranttolevel[id] = P4estTypes.level(quadrant)
    data.quadranttotreeid[id] = treeid
    data.quadranttocoordinate[id, :] .= P4estTypes.coordinates(quadrant)

    return
end

function materializequadrantdata(forest, ghost)
    ghosts = P4estTypes.ghosts(ghost)

    localnumberofquadrants = P4estTypes.lengthoflocalquadrants(forest)
    ghostnumberofquadrants = length(ghosts)
    totalnumberofquadrants = localnumberofquadrants + ghostnumberofquadrants

    quadranttolevel = Array{Int8}(undef, totalnumberofquadrants)
    quadranttotreeid = Array{Int32}(undef, totalnumberofquadrants)
    quadranttocoordinate =
        Array{Int32}(undef, totalnumberofquadrants, P4estTypes.quadrantndims(forest))
    data = (; quadranttolevel, quadranttotreeid, quadranttocoordinate)

    # Fill in information for the local quadrants
    P4estTypes.iterateforest(
        forest;
        ghost,
        userdata = data,
        volume = materializeforestvolumedata,
    )

    # Fill in information for the ghost layer quadrants
    for (quadid, quadrant) in enumerate(ghosts)
        id = quadid + localnumberofquadrants

        quadranttolevel[id] = P4estTypes.level(quadrant)
        quadranttotreeid[id] = P4estTypes.unsafe_which_tree(quadrant)
        quadranttocoordinate[id, :] .= P4estTypes.coordinates(quadrant)
    end

    return (quadranttolevel, quadranttotreeid, quadranttocoordinate)
end

"""
    materializequadranttoglobalid(forest, ghost)

Generate the global ids for quadrants in the `forest` and the `ghost` layer.
"""
function materializequadranttoglobalid(forest, ghost)
    rank = MPI.Comm_rank(forest.comm)

    ghosts = P4estTypes.ghosts(ghost)

    localnumberofquadrants = P4estTypes.lengthoflocalquadrants(forest)
    ghostnumberofquadrants = length(ghosts)
    totalnumberofquadrants = localnumberofquadrants + ghostnumberofquadrants

    GC.@preserve forest ghost begin
        global_first_quadrant = P4estTypes.unsafe_global_first_quadrant(forest)
        gfq = global_first_quadrant[rank+1]
        T = eltype(global_first_quadrant)

        proc_offsets = P4estTypes.unsafe_proc_offsets(ghost)

        globalquadrantids = zeros(T, totalnumberofquadrants)
        for i = 1:localnumberofquadrants
            globalquadrantids[i] = i + gfq
        end
        for i = 1:ghostnumberofquadrants
            globalquadrantids[i+localnumberofquadrants] =
                P4estTypes.unsafe_local_num(ghosts[i])
        end
        for r = 1:length(proc_offsets)-1
            for o = (proc_offsets[r]+1):proc_offsets[r+1]
                globalquadrantids[o+localnumberofquadrants] += global_first_quadrant[r]
            end
        end
    end

    return globalquadrantids
end

function materializedtoc(forest, ghost, nodes, quadrantcommpattern, comm)
    localnumberofquadrants = P4estTypes.lengthoflocalquadrants(forest)
    ghostnumberofquadrants = length(P4estTypes.ghosts(ghost))
    totalnumberofquadrants = localnumberofquadrants + ghostnumberofquadrants

    dtoc_owned = P4estTypes.unsafe_element_nodes(nodes)
    dtoc = zeros(eltype(dtoc_owned), size(dtoc_owned)[1:end-1]..., totalnumberofquadrants)
    dtoc[1:length(dtoc_owned)] .= vec(dtoc_owned)
    dtoc_global = P4estTypes.globalid.(Ref(nodes), dtoc) .+ 0x1

    pattern = expand(quadrantcommpattern, prod(size(dtoc_owned)[1:end-1]))

    cm = commmanager(eltype(dtoc_global), comm, pattern, 0)

    share!(dtoc_global, cm)

    dtoc_local = numbercontiguous(eltype(dtoc_owned), dtoc_global)

    return (dtoc_local, dtoc_global)
end

function materializectod(dtoc)
    dtoc = vec(dtoc)
    data = similar(dtoc, Bool)
    fill!(data, true)
    return sparse(1:length(dtoc), dtoc, data)
end

materializequadranttofacecode(nodes) = copy(P4estTypes.unsafe_face_code(nodes))

function generate(warp::Function, gm::GridManager)
    # Need to get integer coordinates of cells

    A = arraytype(referencecell(gm))

    ghost = P4estTypes.ghostlayer(forest(gm))
    nodes = P4estTypes.lnodes(forest(gm); ghost, degree = 2)
    P4estTypes.expand!(ghost, forest(gm), nodes)

    (quadranttolevel, quadranttotreeid, quadranttocoordinate) =
        materializequadrantdata(forest(gm), ghost)

    quadranttoglobalid = materializequadranttoglobalid(forest(gm), ghost)

    quadrantcommpattern = materializequadrantcommpattern(forest(gm), ghost)

    (dtoc_degree2_local, dtoc_degree2_global) =
        materializedtoc(forest(gm), ghost, nodes, quadrantcommpattern, comm(gm))

    discontinuoustocontinuous =
        materializedtoc(referencecell(gm), dtoc_degree2_local, dtoc_degree2_global)

    continuoustodiscontinuous = materializectod(discontinuoustocontinuous)

    nodecommpattern = materializenodecommpattern(
        referencecell(gm),
        continuoustodiscontinuous,
        quadrantcommpattern,
    )

    parentnodes = materializeparentnodes(
        referencecell(gm),
        continuoustodiscontinuous,
        quadranttoglobalid,
        quadranttolevel,
    )

    quadranttofacecode = materializequadranttofacecode(nodes)

    # Send data to the device
    quadranttolevel = A(pin(A, quadranttolevel))
    quadranttotreeid = A(pin(A, quadranttotreeid))
    quadranttocoordinate = A(pin(A, quadranttocoordinate))
    quadranttofacecode = A(pin(A, quadranttofacecode))
    parentnodes = A(pin(A, parentnodes))
    nodecommpattern = Adapt.adapt(A, nodecommpattern)
    continuoustodiscontinuous = adaptsparse(A, continuoustodiscontinuous)
    discontinuoustocontinuous = Adapt.adapt(A, discontinuoustocontinuous)

    points = materializepoints(
        referencecell(gm),
        coarsegridcells(gm),
        coarsegridvertices(gm),
        quadranttolevel,
        quadranttotreeid,
        quadranttocoordinate,
        forest(gm),
        comm(gm),
    )

    coarsegrid_warp = Raven.warp(coarsegrid(gm))
    points = warp.(coarsegrid_warp.(points))

    volumemetrics, surfacemetrics = materializemetrics(referencecell(gm), points)

    part = MPI.Comm_rank(comm(gm)) + 1
    nparts = MPI.Comm_size(comm(gm))
    GC.@preserve gm begin
        global_first_quadrant = P4estTypes.unsafe_global_first_quadrant(forest(gm))
        offset = global_first_quadrant[part]
    end

    return Grid(
        comm(gm),
        part,
        nparts,
        referencecell(gm),
        offset,
        length(gm),
        points,
        volumemetrics,
        surfacemetrics,
        quadranttolevel,
        quadranttotreeid,
        quadranttofacecode,
        parentnodes,
        nodecommpattern,
        continuoustodiscontinuous,
        discontinuoustocontinuous,
    )
end
