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
isextruded(gm::GridManager) = isextruded(coarsegrid(gm))

function GridManager(
    referencecell,
    coarsegrid;
    comm = MPI.COMM_WORLD,
    min_level = 0,
    fill_uniform = true,
)
    if !MPI.Initialized()
        MPI.Init()
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
    P4estTypes.unsafe_storeuserdata!(
        quadrant,
        QuadData(id, P4estTypes.level(quadrant), flags[id]),
    )
end

function fill_adapt_flags(::P4estTypes.Pxest{X}, _, quadrant, _, _, flags) where {X}
    d = P4estTypes.unsafe_loaduserdata(quadrant, QuadData)

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
    outd = P4estTypes.unsafe_loaduserdata(first(outgoing), QuadData)

    for quadrant in incoming
        P4estTypes.unsafe_storeuserdata!(
            quadrant,
            QuadData(outd.old_id, outd.old_level, AdaptTouched),
        )
    end

    return
end

function refine_quads(_, _, quadrant)
    retval = P4estTypes.unsafe_loaduserdata(quadrant, QuadData).adapt_flag == AdaptRefine
    return retval
end

function coarsen_quads(_, _, children)
    coarsen = true

    for child in children
        coarsen &=
            P4estTypes.unsafe_loaduserdata(child, QuadData).adapt_flag == AdaptCoarsen
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

function extrudequadrantdata(
    unextrudedquadranttolevel,
    unextrudedquadranttotreeid,
    unextrudedquadranttocoordinate,
    columnnumberofquadrants,
)
    quadranttolevel = repeat(unextrudedquadranttolevel, inner = columnnumberofquadrants)
    quadranttotreeid = similar(unextrudedquadranttotreeid, size(quadranttolevel))

    for i = 1:length(unextrudedquadranttotreeid)
        treeidoffset = (unextrudedquadranttotreeid[i] - 1) * columnnumberofquadrants
        for j = 1:columnnumberofquadrants
            quadranttotreeid[(i-1)*columnnumberofquadrants+j] = treeidoffset + j
        end
    end

    u = unextrudedquadranttocoordinate
    q = similar(u, (size(u, 1), size(u, 2) + 1))
    q[:, 1:size(u, 2)] .= u
    q[:, size(u, 2)+1] .= zero(eltype(u))
    quadranttocoordinate = repeat(q, inner = (columnnumberofquadrants, 1))

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

    GC.@preserve forest ghost ghosts begin
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

function extrudequadranttoglobalid(unextrudedquadranttoglobalid, columnnumberofquadrants)
    unextrudednumberofquadrants = length(unextrudedquadranttoglobalid)
    quadranttoglobalid = similar(
        unextrudedquadranttoglobalid,
        unextrudednumberofquadrants * columnnumberofquadrants,
    )

    for q = 1:unextrudednumberofquadrants
        p = unextrudedquadranttoglobalid[q]
        for c = 1:columnnumberofquadrants
            quadranttoglobalid[(q-1)*columnnumberofquadrants+c] =
                (p - 1) * columnnumberofquadrants + c
        end
    end

    return quadranttoglobalid
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

    cm = commmanager(eltype(dtoc_global), pattern; comm)

    share!(dtoc_global, cm)

    dtoc_local = numbercontiguous(eltype(dtoc_owned), dtoc_global)

    return (dtoc_local, dtoc_global)
end

function extrudedtoc(
    unextrudeddtoc::AbstractArray{T,3},
    columnnumberofquadrants,
    columnnumberofcelldofs,
    columnisperiodic,
) where {T}
    s = size(unextrudeddtoc)
    dtoc = similar(
        unextrudeddtoc,
        s[1],
        s[2],
        columnnumberofcelldofs,
        columnnumberofquadrants,
        s[end],
    )

    numccnodes = columnnumberofquadrants * (columnnumberofcelldofs - 1) + !columnisperiodic

    for q = 1:s[end], c = 1:columnnumberofquadrants
        for k = 1:columnnumberofcelldofs
            cnode = (c - 1) * (columnnumberofcelldofs - 1) + k
            @assert cnode <= numccnodes
            for j = 1:s[2], i = 1:s[1]
                dtoc[i, j, k, c, q] = (unextrudeddtoc[i, j, q] - 1) * numccnodes + cnode

            end
        end
        if columnisperiodic
            for j = 1:s[2], i = 1:s[1]
                dtoc[i, j, end, end, q] = dtoc[i, j, 1, 1, q]
            end
        end
    end

    return reshape(
        dtoc,
        s[1],
        s[2],
        columnnumberofcelldofs,
        columnnumberofquadrants * s[end],
    )
end

function materializectod(dtoc)
    dtoc = vec(dtoc)
    data = similar(dtoc, Bool)
    fill!(data, true)
    return sparse(1:length(dtoc), dtoc, data)
end

materializequadranttofacecode(nodes) = copy(P4estTypes.unsafe_face_code(nodes))

function extrudequadranttofacecode(unextrudedquadranttofacecode, columnnumberofquadrants)
    return repeat(unextrudedquadranttofacecode, inner = columnnumberofquadrants)
end

function materializequadrantcommlists(localnumberofquadrants, quadrantcommpattern)
    communicatingcells = unique!(sort(quadrantcommpattern.sendindices))
    noncommunicatingcells = setdiff(0x1:localnumberofquadrants, communicatingcells)

    return (communicatingcells, noncommunicatingcells)
end

function generate(warp::Function, gm::GridManager)
    # Need to get integer coordinates of cells

    A = arraytype(referencecell(gm))

    ghost = P4estTypes.ghostlayer(forest(gm))
    nodes = P4estTypes.lnodes(forest(gm); ghost, degree = 3)
    P4estTypes.expand!(ghost, forest(gm), nodes)

    localnumberofquadrants = P4estTypes.lengthoflocalquadrants(forest(gm))

    (quadranttolevel, quadranttotreeid, quadranttocoordinate) =
        materializequadrantdata(forest(gm), ghost)

    quadranttoglobalid = materializequadranttoglobalid(forest(gm), ghost)

    quadrantcommpattern = materializequadrantcommpattern(forest(gm), ghost)

    (dtoc_degree3_local, dtoc_degree3_global) =
        materializedtoc(forest(gm), ghost, nodes, quadrantcommpattern, comm(gm))

    quadranttofacecode = materializequadranttofacecode(nodes)

    if isextruded(coarsegrid(gm))
        columnnumberofquadrants = columnlength(coarsegrid(gm))

        localnumberofquadrants *= columnnumberofquadrants

        (quadranttolevel, quadranttotreeid, quadranttocoordinate) = extrudequadrantdata(
            quadranttolevel,
            quadranttotreeid,
            quadranttocoordinate,
            columnnumberofquadrants,
        )

        quadranttoglobalid =
            extrudequadranttoglobalid(quadranttoglobalid, columnnumberofquadrants)

        quadrantcommpattern = expand(quadrantcommpattern, columnnumberofquadrants)

        dtoc_degree3_local = extrudedtoc(
            dtoc_degree3_local,
            columnnumberofquadrants,
            4,
            last(isperiodic(coarsegrid(gm))),
        )

        dtoc_degree3_global = extrudedtoc(
            dtoc_degree3_global,
            columnnumberofquadrants,
            4,
            last(isperiodic(coarsegrid(gm))),
        )

        quadranttofacecode =
            extrudequadranttofacecode(quadranttofacecode, columnnumberofquadrants)
    end

    discontinuoustocontinuous =
        materializedtoc(referencecell(gm), dtoc_degree3_local, dtoc_degree3_global)

    ctod_degree3_local = materializectod(dtoc_degree3_local)

    facemaps, quadranttoboundary = materializefacemaps(
        referencecell(gm),
        localnumberofquadrants,
        ctod_degree3_local,
        dtoc_degree3_local,
        dtoc_degree3_global,
        quadranttolevel,
        quadranttoglobalid,
    )

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

    communicatingquadrants, noncommunicatingquadrants =
        materializequadrantcommlists(localnumberofquadrants, quadrantcommpattern)

    # Send data to the device
    quadranttolevel = A(pin(A, quadranttolevel))
    quadranttotreeid = A(pin(A, quadranttotreeid))
    quadranttocoordinate = A(pin(A, quadranttocoordinate))
    quadranttofacecode = A(pin(A, quadranttofacecode))
    quadranttoboundary = A(pin(A, quadranttoboundary))
    parentnodes = A(pin(A, parentnodes))
    nodecommpattern = Adapt.adapt(A, nodecommpattern)
    continuoustodiscontinuous = adaptsparse(A, continuoustodiscontinuous)
    discontinuoustocontinuous = Adapt.adapt(A, discontinuoustocontinuous)
    communicatingquadrants = Adapt.adapt(A, communicatingquadrants)
    noncommunicatingquadrants = Adapt.adapt(A, noncommunicatingquadrants)
    facemaps = Adapt.adapt(A, facemaps)

    if coarsegrid(gm) isa AbstractBrickGrid
        points = materializebrickpoints(
            referencecell(gm),
            coarsegridcells(gm),
            coarsegridvertices(gm),
            quadranttolevel,
            quadranttotreeid,
            quadranttocoordinate,
            localnumberofquadrants,
            comm(gm),
        )
    elseif coarsegrid(gm) isa MeshImportCoarseGrid
        meshimport = Raven.meshimport(coarsegrid(gm))
        quadranttointerpolation = materializequadranttointerpolation(meshimport)
        quadranttointerpolation = A(pin(A, quadranttointerpolation))
        faceinterpolation = interpolation(meshimport)
        faceinterpolation = collect(
            transpose(
                reinterpret(reshape, eltype(eltype(faceinterpolation)), faceinterpolation),
            ),
        )
        faceinterpolation = A(pin(A, faceinterpolation))
        points = materializepoints(
            referencecell(gm),
            coarsegridcells(gm),
            coarsegridvertices(gm),
            interpolationdegree(meshimport),
            faceinterpolation,
            quadranttointerpolation,
            quadranttolevel,
            quadranttotreeid,
            quadranttocoordinate,
            localnumberofquadrants,
            comm(gm),
        )
    else
        points = materializepoints(
            referencecell(gm),
            coarsegridcells(gm),
            coarsegridvertices(gm),
            quadranttolevel,
            quadranttotreeid,
            quadranttocoordinate,
            localnumberofquadrants,
            comm(gm),
        )
    end

    coarsegrid_warp = Raven.warp(coarsegrid(gm))
    points = warp.(coarsegrid_warp.(points))

    fillghosts!(points, fill(NaN, eltype(points)))
    pcm = commmanager(eltype(points), nodecommpattern; comm = comm(gm))
    share!(points, pcm)

    isunwarpedbrick = coarsegrid(gm) isa AbstractBrickGrid && warp == identity

    volumemetrics, surfacemetrics = materializemetrics(
        referencecell(gm),
        points,
        facemaps,
        comm(gm),
        nodecommpattern,
        isunwarpedbrick,
    )

    part = MPI.Comm_rank(comm(gm)) + 1
    nparts = MPI.Comm_size(comm(gm))
    GC.@preserve gm begin
        global_first_quadrant = P4estTypes.unsafe_global_first_quadrant(forest(gm))
        offset = global_first_quadrant[part]
    end

    if isextruded(coarsegrid(gm))
        columnnumberofquadrants = columnlength(coarsegrid(gm))
        offset *= columnnumberofquadrants
    end

    return Grid(
        comm(gm),
        part,
        nparts,
        referencecell(gm),
        offset,
        convert(Int, localnumberofquadrants),
        points,
        volumemetrics,
        surfacemetrics,
        quadranttolevel,
        quadranttotreeid,
        quadranttofacecode,
        quadranttoboundary,
        parentnodes,
        nodecommpattern,
        continuoustodiscontinuous,
        discontinuoustocontinuous,
        communicatingquadrants,
        noncommunicatingquadrants,
        facemaps,
    )
end


function materializequadranttointerpolation(abaqus::AbaqusMeshImport)
    dims = length(abaqus.connectivity[1]) == 4 ? 2 : 3
    nodesperface = (abaqus.face_degree + 1)^(dims - 1)
    numberofelements = length(abaqus.face_iscurved)
    facesperelement = length(abaqus.face_iscurved[1])
    quadranttointerpolation = Array{Int32}(undef, numberofelements, facesperelement)
    idx = 1
    for element = 1:numberofelements
        for face = 1:facesperelement
            if abaqus.face_iscurved[element][face] == 1
                quadranttointerpolation[element, face] = idx
                idx += nodesperface
            else
                quadranttointerpolation[element, face] = 0
            end
        end
    end

    return quadranttointerpolation
end

function Base.show(io::IO, g::GridManager)
    compact = get(io, :compact, false)
    print(io, "GridManager(")
    show(io, referencecell(g))
    print(io, ", ")
    show(io, coarsegrid(g))
    print(io, "; comm=")
    show(io, comm(g))
    print(io, ")")
    if !compact
        nlocal = P4estTypes.lengthoflocalquadrants(forest(g))
        nglobal = P4estTypes.lengthofglobalquadrants(forest(g))
        print(io, " with $nlocal of the $nglobal global elements")
    end

    return
end

function Base.showarg(io::IO, g::GridManager, toplevel)
    !toplevel && print(io, "::")

    print(io, "GridManager{")
    Base.showarg(io, referencecell(g), false)
    print(io, ", ")
    Base.showarg(io, coarsegrid(g), false)
    print(io, "}")

    if toplevel
        nlocal = P4estTypes.lengthoflocalquadrants(forest(g))
        nglobal = P4estTypes.lengthofglobalquadrants(forest(g))
        print(io, " with $nlocal of the $nglobal global elements")
    end

    return
end

Base.summary(io::IO, g::GridManager) = Base.showarg(io, g, true)
