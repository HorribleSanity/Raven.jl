using MPI, P4estTypes;
MPI.Init();
using SparseArrays

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

function getelementcomm(forest, ghost)
    ms = mirrors(ghost)
    gs = ghosts(ghost)

    GC.@preserve ghost begin
        mirror_proc_mirrors = P4estTypes.unsafe_mirror_proc_mirrors(ghost)
        mirror_proc_offsets = P4estTypes.unsafe_mirror_proc_offsets(ghost)

        num_local = lengthoflocalquadrants(forest)
        T = typeof(num_local)
        num_ghosts = T(length(gs))

        sendindices = similar(mirror_proc_mirrors, T)
        for i in eachindex(sendindices)
            sendindices[i] = P4estTypes.unsafe_local_num(ms[mirror_proc_mirrors[i]+1])
        end
        (sendranks, sendrankindices) = _offsets_to_ranges(mirror_proc_offsets)
        send = (; indices = sendindices, ranks = sendranks, rankindices = sendrankindices)

        proc_offsets = P4estTypes.unsafe_proc_offsets(ghost)

        recvindices = (num_local+0x1):(num_local+num_ghosts)
        (recvranks, recvrankindices) = _offsets_to_ranges(proc_offsets)

        recv = (; indices = recvindices, ranks = recvranks, rankindices = recvrankindices)
    end

    return (; recv, send)
end

function getdims(celldims, dtoc_degree2_global, node, elem)
    dims = ntuple(length(celldims)) do n
        dim =
            node[n] == 2 ? (2:celldims[n]-1) :
            node[n] == 1 ? (1:1) : (celldims[n]:celldims[n])
        shift = ntuple(m -> m == n ? 1 : 0, length(celldims))

        # Flip the dimension to match the orientation of the degree 2 node numbering
        if node[n] == 2 &&
           dtoc_degree2_global[(node .+ shift)..., elem] <
           dtoc_degree2_global[(node .- shift)..., elem]
            dim = reverse(dim)
        end

        return StepRange(dim)
    end

    return dims
end

function getdtoc(celldims, dtoc_degree2, dtoc_degree2_global)
    # Compute the offsets for the cell node numbering
    offsets = zeros(Int, maximum(dtoc_degree2) + 1)
    for i in eachindex(IndexCartesian(), dtoc_degree2)
        l = dtoc_degree2[i]
        I = Tuple(i)
        node = I[1:end-1]

        if sum(node .== 2) == 0
            # corner
            offsets[l+1] = 1
        else
            # volume, face, edge
            offsets[l+1] =
                prod(ntuple(n -> node[n] == 2 ? (celldims[n] - 2) : 1, length(node)))
        end
    end
    cumsum!(offsets, offsets)

    dtoc = zeros(Int, celldims..., last(size(dtoc_degree2)))
    for i in eachindex(IndexCartesian(), dtoc_degree2)
        l = dtoc_degree2[i]
        I = Tuple(i)
        node = I[1:end-1]
        elem = I[end]
        dims = getdims(celldims, dtoc_degree2_global, node, elem)
        for (j, k) in enumerate(CartesianIndices(dims))
            dtoc[k, elem] = j + offsets[l]
        end
    end

    return dtoc
end

function share!(array, forest, ghost)
    elemcomm = getelementcomm(forest, ghost)

    nelems = last(size(array))
    matrix = reshape(array, :, nelems)

    recvreqs = MPI.MultiRequest(length(elemcomm.recv.ranks))
    for r in eachindex(elemcomm.recv.ranks)
        elems = elemcomm.recv.indices[elemcomm.recv.rankindices[r]]
        data = @view matrix[:, elems]
        # @info "Recv $rank <-- $(elemcomm.recv.ranks[r])  $elems $(size(data))"
        MPI.Irecv!(data, elemcomm.recv.ranks[r], 666, forest.comm, recvreqs[r])
    end

    sendreqs = MPI.MultiRequest(length(elemcomm.send.ranks))
    for r in eachindex(elemcomm.send.ranks)
        elems = elemcomm.send.indices[elemcomm.send.rankindices[r]]
        data = matrix[:, elems]
        # @info "Send $rank --> $(elemcomm.send.ranks[r]) $elems $(size(data))"
        MPI.Isend(data, elemcomm.send.ranks[r], 666, forest.comm, sendreqs[r])
    end

    MPI.Waitall(recvreqs)
    MPI.Waitall(sendreqs)
end

#function localnumbering(::Type{T}, dtoc_global) where {T}
#    d = Dict{eltype(dtoc_global), T}()
#
#    for g in dtoc_global
#        push!(d, g => one(T))
#    end
#
#    for (l, g) in enumerate(sort!(collect(keys(d))))
#        pop!(d, g)
#        push!(d, g => l)
#    end
#
#    dtoc_local = get.(Ref(d), dtoc_global, -1)
#
#    return dtoc_local
#end

function numbercontiguous(::Type{T}, A; by = identity) where {T}
    p = sortperm(vec(A); by = by)
    notequalprevious = fill!(similar(p, Bool), false)

    for i in Iterators.drop(eachindex(p), 1)
        notequalprevious[i] = by(A[p[i]]) != by(A[p[i-1]])
    end

    B = similar(A, T)
    B[p] .= cumsum(notequalprevious) .+ 1

    return B
end

# function getcelldofs(celldims)
#     celldofs = Array{Array{Int}}(undef, ntuple(_->3, length(celldims)))
#
#     li = LinearIndices(celldims)
#
#     for i in eachindex(IndexCartesian(), celldofs)
#         node = Tuple(i)
#         indices = ntuple(length(celldims)) do n
#             if node[n] == 2
#                 return Colon()
#             elseif node[n] == 1
#                 return 1
#             else
#                 return celldims[n]
#             end
#         end
#
#         celldofs[i] = @view li[indices...]
#     end
#
#     return celldofs
# end

@inline function elemview(array, elems)
    return view(array, ntuple(_ -> Colon(), ndims(array) - 1)..., elems)
end

function getdofcomm(celldims, dtoc_degree2_global, forest, ghost, nodes)
    rank = MPI.Comm_rank(forest.comm)
    Np = prod(celldims)
    li = LinearIndices(celldims)

    s = sharers(nodes)
    owned = elemview(dtoc_degree2_global, 1:lengthoflocalquadrants(forest))
    senddofs = Dict{keytype(s),Array{Int}}()
    for i in eachindex(IndexCartesian(), owned)
        g = owned[i]
        I = Tuple(i)
        node = I[1:end-1]
        elem = I[end]

        for (k, v) in zip(keys(s), values(s))
            if k != rank && g ∈ v
                dofs = get!(senddofs, k) do
                    Int[]
                end
                dims = getdims(celldims, dtoc_degree2_global, node, elem)
                # @info "$rank --> $k : $g $(li[dims...] .+ (elem-1) * Np) :: $(dims) $node $elem"
                for dof in li[dims...]
                    push!(dofs, dof + (elem - 1) * Np)
                end
            end
        end
    end
    # @info senddofs

    sendranks = sort!(collect(keys(senddofs)))
    sendindices = Int[]
    sendrankindices = UnitRange{Int}[]
    sendoffset = 0
    for r in sendranks
        dofs = senddofs[r]
        append!(sendindices, dofs)
        push!(sendrankindices, (1:length(dofs)) .+ sendoffset)

        sendoffset += length(dofs)
    end

    send = (indices = sendindices, ranks = sendranks, rankindices = sendrankindices)

    elemcomm = getelementcomm(forest, ghost)
    recvdofs = Vector{Vector{Int}}(undef, length(elemcomm.recv.ranks))
    for (j, offset) in enumerate(elemcomm.recv.rankindices)
        elems = elemcomm.recv.indices[offset]
        remote = elemview(dtoc_degree2_global, elems)
        v = s[elemcomm.recv.ranks[j]]

        dofs = Int[]

        for i in eachindex(IndexCartesian(), remote)
            g = remote[i]
            I = Tuple(i)
            node = I[1:end-1]
            elem = elems[I[end]]

            if g ∈ v
                dims = getdims(celldims, dtoc_degree2_global, node, elem)
                # @info "$rank <-- $(elemcomm.recv.ranks[j]) : $g $(li[dims...] .+ (elem-1) * Np) :: $(dims) $node $elem"
                for dof in li[dims...]
                    push!(dofs, dof + (elem - 1) * Np)
                end
            end
        end

        recvdofs[j] = dofs
    end
    # @info recvdofs

    recvranks = elemcomm.recv.ranks
    recvindices = Int[]
    recvrankindices = UnitRange{Int}[]
    recvoffset = 0
    for i = 1:length(recvdofs)
        dofs = recvdofs[i]
        append!(recvindices, dofs)
        push!(recvrankindices, (1:length(dofs)) .+ recvoffset)

        recvoffset += length(dofs)
    end

    recv = (indices = recvindices, ranks = recvranks, rankindices = recvrankindices)

    return (; recv, send)
end

function sharedofs!(array, comm, dofcomm)
    senddata = array[dofcomm.send.indices]
    recvdata = similar(array, size(dofcomm.recv.indices))
    # rank = MPI.Comm_rank(comm)

    recvreqs = MPI.MultiRequest(length(dofcomm.recv.ranks))
    for r in eachindex(dofcomm.recv.ranks)
        data = view(recvdata, dofcomm.recv.rankindices[r])
        # @info "Recv $rank <-- $(dofcomm.recv.ranks[r]) $(size(data)) $(dofcomm.recv.rankindices[r])"
        MPI.Irecv!(data, dofcomm.recv.ranks[r], 777, comm, recvreqs[r])
    end

    sendreqs = MPI.MultiRequest(length(dofcomm.send.ranks))
    for r in eachindex(dofcomm.send.ranks)
        data = view(senddata, dofcomm.send.rankindices[r])
        # @info "Send $rank --> $(dofcomm.send.ranks[r]) $(size(data)) $(dofcomm.send.rankindices[r])"
        MPI.Isend(data, dofcomm.send.ranks[r], 777, comm, sendreqs[r])
    end

    MPI.Waitall(recvreqs)
    MPI.Waitall(sendreqs)

    array[dofcomm.recv.indices] .= recvdata
end

function getglobalelementids(forest, ghost)
    rank = MPI.Comm_rank(forest.comm)

    gs = ghosts(ghost)
    nelem_owned = lengthoflocalquadrants(forest)
    nelem_ghost = length(gs)
    nelem = nelem_owned + nelem_ghost

    GC.@preserve forest ghost begin
        global_first_quadrant = P4estTypes.unsafe_global_first_quadrant(forest)
        gfq = global_first_quadrant[rank+1]
        T = eltype(global_first_quadrant)

        proc_offsets = P4estTypes.unsafe_proc_offsets(ghost)

        globalelementids = zeros(T, nelem)
        for i = 1:nelem_owned
            globalelementids[i] = i + gfq
        end
        for i = 1:nelem_ghost
            globalelementids[i+nelem_owned] = P4estTypes.unsafe_local_num(gs[i])
        end
        for r = 1:length(proc_offsets)-1
            for o = (proc_offsets[r]+1):proc_offsets[r+1]
                globalelementids[o+nelem_owned] += global_first_quadrant[r]
            end
        end
    end

    return globalelementids
end

function getelementlevels(forest, ghost)
    gs = ghosts(ghost)
    nelem_owned = lengthoflocalquadrants(forest)
    nelem_ghost = length(gs)
    nelem = nelem_owned + nelem_ghost

    levels = zeros(Int8, nelem)
    for (i, q) in enumerate(Iterators.flatten(forest))
        levels[i] = level(q)
    end
    for (i, q) in enumerate(gs)
        levels[i+nelem_owned] = level(q)
    end

    return levels
end

function getparentdofs(ctod, celldims, forest, ghost)
    globalelementids = getglobalelementids(forest, ghost)
    levels = getelementlevels(forest, ghost)

    Np = prod(celldims)
    rows = rowvals(ctod)
    m, n = size(ctod)
    parentdofs = zeros(eltype(globalelementids), m)
    for j = 1:n
        level = typemax(Int8)
        gid = typemax(eltype(globalelementids))
        pdof = 0
        for ii in nzrange(ctod, j)
            i = rows[ii]
            e = cld(i, Np)
            if levels[e] ≤ level && globalelementids[e] < gid
                level = levels[e]
                gid = globalelementids[e]
                pdof = i
            end
        end
        @assert pdof != 0
        for ii in nzrange(ctod, j)
            i = rows[ii]
            parentdofs[i] = pdof
        end
    end

    return reshape(parentdofs, celldims..., :)
end

let
    celldims = (4, 3)
    forest = pxest(brick(2, 2))

    refine!(forest; refine = (_, tid, _) -> tid == 4)

    ghost = ghostlayer(forest)
    nodes = lnodes(forest; ghost, degree = 2)
    expand!(ghost, forest, nodes)

    nelem_owned = lengthoflocalquadrants(forest)
    nelem_ghost = length(ghosts(ghost))
    nelem = nelem_owned + nelem_ghost

    dtoc_degree2_owned = P4estTypes.unsafe_element_nodes(nodes)
    dtoc_degree2 =
        zeros(eltype(dtoc_degree2_owned), size(dtoc_degree2_owned)[1:end-1]..., nelem)
    dtoc_degree2[1:length(dtoc_degree2_owned)] .= vec(dtoc_degree2_owned)
    dtoc_degree2_global = globalid.(Ref(nodes), dtoc_degree2)
    share!(dtoc_degree2_global, forest, ghost)
    dtoc_degree2_local = numbercontiguous(eltype(dtoc_degree2_owned), dtoc_degree2_global)
    #@info "DToC global" dtoc_degree2_global
    #@info "DToC local" dtoc_degree2_local
    #@info "sharers" sharers_degree2
    #
    #

    face_codes_owned = P4estTypes.unsafe_face_code(nodes)
    face_codes = zeros(eltype(face_codes_owned), nelem)
    face_codes[1:length(face_codes_owned)] .= face_codes_owned
    share!(face_codes, forest, ghost)

    dtoc = getdtoc(celldims, dtoc_degree2_local, dtoc_degree2_global)
    dofcomm = getdofcomm(celldims, dtoc_degree2_global, forest, ghost, nodes)
    rank = MPI.Comm_rank(forest.comm)

    ctod = sparse(1:length(dtoc), vec(dtoc), ones(Bool, length(dtoc)))
    @assert ctod * collect(1:size(ctod, 2)) == vec(dtoc)


    # @descend getdtoc(celldims, dtoc_degree2_local, dtoc_degree2_global)
    # @descend getdofcomm(celldims, dtoc_degree2_global, forest, ghost, nodes)
    # @descend getparentdofs(ctod, celldims, forest, ghost)

    parentdofs = getparentdofs(ctod, celldims, forest, ghost)

    elemcomm = getelementcomm(forest, ghost)

    @info "Rank $rank" celldims forest dtoc
    @info "ctod" ctod
    @info "parent degrees of freedom" parentdofs
    @info "comms" elemcomm dofcomm
    @info sharers(nodes)

    #dtoccommed = copy(dtoc)
    #dtoccommed[dofcomm.recv.indices] .= 0
    #sharedofs!(dtoccommed, forest.comm, dofcomm)

end
