const COMM_MANAGERS = WeakRef[]

struct CommPattern{AT,RI,RR,RRI,SI,SR,SRI}
    recvindices::RI
    recvranks::RR
    recvrankindices::RRI
    sendindices::SI
    sendranks::SR
    sendrankindices::SRI
end
function CommPattern{AT}(
    recvindices::RI,
    recvranks::RR,
    recvrankindices::RRI,
    sendindices::SI,
    sendranks::SR,
    sendrankindices::SRI,
) where {AT,RI,RR,RRI,SI,SR,SRI}
    return CommPattern{AT,RI,RR,RRI,SI,SR,SRI}(
        recvindices,
        recvranks,
        recvrankindices,
        sendindices,
        sendranks,
        sendrankindices,
    )
end

arraytype(::CommPattern{AT}) where {AT} = AT

function Adapt.adapt_structure(to, obj::CommPattern)
    CommPattern{to}(
        adapt(to, obj.recvindices),
        obj.recvranks,
        obj.recvrankindices,
        adapt(to, obj.sendindices),
        obj.sendranks,
        obj.sendrankindices,
    )
end

function expand(r::UnitRange, factor)
    a = (first(r) - 0x1) * factor + 0x1
    b = last(r) * factor
    return a:b
end

@kernel function expand_vector_kernel!(dest, src, factor, offset)
    i = @index(Global)

    @inbounds begin
        b, a = fldmod1(src[i], offset)

        for f = 1:factor
            dest[f+(i-0x1)*factor] = a + (f - 0x1) * offset + (b - 0x1) * factor * offset
        end
    end
end

function expand(v::AbstractVector, factor, offset = 1)
    w = similar(v, length(v) * factor)

    expand_vector_kernel!(get_backend(w), 256)(w, v, factor, offset, ndrange = length(v))

    return w
end

"""
    expand(pattern::CommPattern, factor, offset)

Create a new `CommPattern` where the `recvindices` and `sendindices` are
expanded in size by `factor` entries and offset by `offset`.

For example, to expand a `nodecommpattern` to communicate all fields of an
array that is indexed via `(node, field, element)` use
`expand(nodecommpattern(grid), nfields, nnodes)`.
"""
function expand(pattern::CommPattern{AT}, factor, offset = 1) where {AT}
    recvindices = expand(pattern.recvindices, factor, offset)
    recvranks = copy(pattern.recvranks)
    recvrankindices = similar(pattern.recvrankindices)
    @assert eltype(recvrankindices) <: UnitRange
    for i in eachindex(recvrankindices)
        recvrankindices[i] = expand(pattern.recvrankindices[i], factor)
    end

    sendindices = expand(pattern.sendindices, factor, offset)
    sendranks = copy(pattern.sendranks)
    sendrankindices = similar(pattern.sendrankindices)
    @assert eltype(sendrankindices) <: UnitRange
    for i in eachindex(sendrankindices)
        sendrankindices[i] = expand(pattern.sendrankindices[i], factor)
    end

    return CommPattern{AT}(
        recvindices,
        recvranks,
        recvrankindices,
        sendindices,
        sendranks,
        sendrankindices,
    )
end

abstract type AbstractCommManager end

mutable struct CommManagerBuffered{CP,RBD,RB,SBD,SB} <: AbstractCommManager
    comm::MPI.Comm
    pattern::CP
    tag::Cint
    recvbufferdevice::RBD
    recvbuffers::RB
    recvrequests::MPI.UnsafeMultiRequest
    sendbufferdevice::SBD
    sendbuffers::SB
    sendrequests::MPI.UnsafeMultiRequest
end

mutable struct CommManagerTripleBuffered{CP,RBC,RBH,RBD,RB,RS,SBC,SBH,SBD,SB,SS} <:
               AbstractCommManager
    comm::MPI.Comm
    pattern::CP
    tag::Cint
    recvbuffercomm::RBC
    recvbufferhost::RBH
    recvbufferdevice::RBD
    recvbufferes::RB
    recvrequests::MPI.UnsafeMultiRequest
    recvstream::RS
    sendbuffercomm::SBC
    sendbufferhost::SBH
    sendbufferdevice::SBD
    sendbuffers::SB
    sendrequests::MPI.UnsafeMultiRequest
    sendstream::SS
end

function _get_mpi_buffers(buffer, rankindices)
    # Hack to make the element type of the buffer arrays concrete
    @assert eltype(rankindices) == UnitRange{eltype(eltype(rankindices))}
    T = typeof(view(buffer, 1:length(rankindices)))

    bufs = Array{MPI.Buffer{T}}(undef, length(rankindices))
    for i in eachindex(rankindices)
        bufs[i] = MPI.Buffer(view(buffer, rankindices[i]))
    end
    return bufs
end

usetriplebuffer(::Type{Array}) = false

function commmanager(T, pattern; kwargs...)
    AT = arraytype(pattern)
    triplebuffer = usetriplebuffer(AT)
    commmanager(T, pattern, Val(triplebuffer); kwargs...)
end

function commmanager(
    T,
    pattern,
    ::Val{triplebuffer};
    comm = MPI.COMM_WORLD,
    tag = 0,
) where {triplebuffer}
    AT = arraytype(pattern)

    if tag < 0 || tag > 32767
        throw(ArgumentError("The tag=$tag is not in the valid range 0:32767"))
    end
    ctag = convert(Cint, tag)

    recvsize = size(pattern.recvindices)
    sendsize = size(pattern.sendindices)

    recvbufferdevice = AT{T}(undef, recvsize)
    sendbufferdevice = AT{T}(undef, sendsize)

    recvrequests = MPI.UnsafeMultiRequest(length(pattern.recvranks))
    sendrequests = MPI.UnsafeMultiRequest(length(pattern.sendranks))

    if triplebuffer
        recvbufferhost = Array{T}(undef, recvsize)
        sendbufferhost = Array{T}(undef, sendsize)

        recvbuffercomm = similar(recvbufferhost)
        sendbuffercomm = similar(sendbufferhost)
    else
        recvbuffercomm = recvbufferdevice
        sendbuffercomm = sendbufferdevice
    end

    recvbuffers = _get_mpi_buffers(recvbuffercomm, pattern.recvrankindices)
    sendbuffers = _get_mpi_buffers(sendbuffercomm, pattern.sendrankindices)

    for i in eachindex(pattern.recvranks)
        #@info "Recv" pattern.recvranks[i] pattern.recvrankindices[i] recvbuffers[i]
        MPI.Recv_init(
            recvbuffers[i],
            comm,
            recvrequests[i];
            source = pattern.recvranks[i],
            tag = ctag,
        )
    end

    for i in eachindex(pattern.sendranks)
        #@info "Send" pattern.sendranks[i] pattern.sendrankindices[i] sendbuffers[i]
        MPI.Send_init(
            sendbuffers[i],
            comm,
            sendrequests[i];
            dest = pattern.sendranks[i],
            tag = ctag,
        )
    end

    cm = if triplebuffer
        backend = get_backend(arraytype(pattern))
        recvstream = Stream(backend)
        sendstream = Stream(backend)
        CommManagerTripleBuffered(
            comm,
            pattern,
            ctag,
            recvbuffercomm,
            recvbufferhost,
            recvbufferdevice,
            recvbuffers,
            recvrequests,
            recvstream,
            sendbuffercomm,
            sendbufferhost,
            sendbufferdevice,
            sendbuffers,
            sendrequests,
            sendstream,
        )
    else
        CommManagerBuffered(
            comm,
            pattern,
            ctag,
            recvbufferdevice,
            recvbuffers,
            recvrequests,
            sendbufferdevice,
            sendbuffers,
            sendrequests,
        )
    end

    finalizer(cm) do cm
        for reqs in (cm.recvrequests, cm.sendrequests)
            for req in reqs
                MPI.free(req)
            end
        end
    end

    push!(COMM_MANAGERS, WeakRef(cm))

    return cm
end

get_backend(cm::AbstractCommManager) = get_backend(arraytype(cm.pattern))

@kernel function setbuffer_kernel!(buffer, src, Is)
    i = @index(Global)
    @inbounds buffer[i] = src[Is[i]]
end

function setbuffer!(buffer, src, Is)
    axes(buffer) == axes(Is) || Broadcast.throwdm(axes(buffer), axes(Is))
    isempty(buffer) && return

    setbuffer_kernel!(get_backend(buffer), 256)(buffer, src, Is, ndrange = length(buffer))

    return
end

@kernel function getbuffer_kernel!(dest, buffer, Is)
    i = @index(Global)
    @inbounds dest[Is[i]] = buffer[i]
end

function getbuffer!(dest, buffer, Is)
    axes(buffer) == axes(Is) || Broadcast.throwdm(axes(buffer), axes(Is))
    isempty(buffer) && return

    getbuffer_kernel!(get_backend(buffer), 256)(dest, buffer, Is, ndrange = length(buffer))

    return
end

function progress(::AbstractCommManager)
    MPI.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, MPI.COMM_WORLD)

    return
end

function start!(A, cm::CommManagerBuffered)
    if !isempty(cm.sendrequests)
        setbuffer!(cm.sendbufferdevice, A, cm.pattern.sendindices)
    end

    if !isempty(cm.sendrequests) || !isempty(cm.recvrequests)
        KernelAbstractions.synchronize(get_backend(cm))
    end

    if !isempty(cm.recvrequests)
        MPI.Startall(cm.recvrequests)
    end

    if !isempty(cm.sendrequests)
        MPI.Startall(cm.sendrequests)
    end

    return
end

function finish!(A, cm::CommManagerBuffered)
    if !isempty(cm.recvrequests)
        MPI.Waitall(cm.recvrequests)
    end

    if !isempty(cm.sendrequests)
        MPI.Waitall(cm.sendrequests)
    end

    if !isempty(cm.recvrequests)
        A = viewwithghosts(A)
        getbuffer!(A, cm.recvbufferdevice, cm.pattern.recvindices)
    end

    return
end

function start!(A, cm::CommManagerTripleBuffered)
    # We use two host buffers each for send and receive.  We do this to have
    # one buffer pinned by the device stack and one pinned for the network
    # interface.  We see deadlocks if two separate buffers are not used.  See,
    # <https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/> for more
    # details.

    if !isempty(cm.recvrequests)
        MPI.Startall(cm.recvrequests)
    end

    if !isempty(cm.sendrequests)
        backend = get_backend(cm)

        # Wait for kernels on the main stream to finish before launching
        # kernels on on different streams.
        KernelAbstractions.synchronize(backend)

        stream!(backend, cm.sendstream) do
            setbuffer!(cm.sendbufferdevice, A, cm.pattern.sendindices)
            KernelAbstractions.copyto!(backend, cm.sendbufferhost, cm.sendbufferdevice)
        end
    end

    return
end

function finish!(A, cm::CommManagerTripleBuffered)
    if !isempty(cm.sendrequests)
        backend = get_backend(cm)
        synchronize(backend, cm.sendstream)
        copyto!(cm.sendbuffercomm, cm.sendbufferhost)
        MPI.Startall(cm.sendrequests)
    end

    if !isempty(cm.recvrequests)
        MPI.Waitall(cm.recvrequests)
        copyto!(cm.recvbufferhost, cm.recvbuffercomm)
        stream!(backend, cm.recvstream) do
            KernelAbstractions.copyto!(backend, cm.recvbufferdevice, cm.recvbufferhost)
            getbuffer!(viewwithghosts(A), cm.recvbufferdevice, cm.pattern.recvindices)
            KernelAbstractions.synchronize(backend)
        end
    end

    if !isempty(cm.sendrequests)
        MPI.Waitall(cm.sendrequests)
    end

    return
end

function share!(A, cm::AbstractCommManager)
    start!(A, cm)

    progress(cm)

    finish!(A, cm)

    return
end
