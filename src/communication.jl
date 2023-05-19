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
        Adapt.adapt(AT, recvindices),
        recvranks,
        recvrankindices,
        Adapt.adapt(AT, sendindices),
        sendranks,
        sendrankindices,
    )
end

arraytype(::CommPattern{AT}) where {AT} = AT

abstract type AbstractCommManager end

struct CommManagerBuffered{CP,RBD,RB,SBD,SB} <: AbstractCommManager
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

struct CommManagerTripleBuffered{CP,RBC,RBH,RBD,RB,SBC,SBH,SBD,SB} <: AbstractCommManager
    comm::MPI.Comm
    pattern::CP
    tag::Cint
    recvbuffercomm::RBC
    recvbufferhost::RBH
    recvbufferdevice::RBD
    recvbufferes::RB
    recvrequests::MPI.UnsafeMultiRequest
    recvtask::Base.RefValue{Task}
    sendbuffercomm::SBC
    sendbufferhost::SBH
    sendbufferdevice::SBD
    sendbuffers::SB
    sendrequests::MPI.UnsafeMultiRequest
    sendtask::Base.RefValue{Task}
end

function _get_mpi_buffers(buffer, rankindices)
    # Hack to make the element type of the buffer arrays concrete
    @assert eltype(rankindices) == typeof(1:1)
    T = typeof(view(buffer, 1:1))

    bufs = Array{MPI.Buffer{T}}(undef, length(rankindices))
    for i in eachindex(rankindices)
        bufs[i] = MPI.Buffer(view(buffer, rankindices[i]))
    end
    return bufs
end

usetriplebuffer(::Type{Array}) = false

function commmanager(T, comm, pattern, tag)
    AT = arraytype(pattern)
    triplebuffer = usetriplebuffer(AT)
    commmanager(T, comm, pattern, tag, Val(triplebuffer))
end

function commmanager(T, comm, pattern, tag, ::Val{triplebuffer}) where {triplebuffer}
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

    return if triplebuffer
        recvtask = Ref(Task(() -> nothing))
        sendtask = Ref(Task(() -> nothing))

        CommManagerTripleBuffered(
            comm,
            pattern,
            ctag,
            recvbuffercomm,
            recvbufferhost,
            recvbufferdevice,
            recvbuffers,
            recvrequests,
            recvtask,
            sendbuffercomm,
            sendbufferhost,
            sendbufferdevice,
            sendbuffers,
            sendrequests,
            sendtask,
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
end

get_backend(cm::AbstractCommManager) = get_backend(arraytype(cm.pattern))

function progress(::AbstractCommManager)
    MPI.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, MPI.COMM_WORLD)
end

function start!(A, cm::CommManagerBuffered)
    cm.sendbufferdevice .= A[cm.pattern.sendindices]
    KernelAbstractions.synchronize(get_backend(cm))

    MPI.Startall(cm.recvrequests)
    MPI.Startall(cm.sendrequests)
end

function finish!(A, cm::CommManagerBuffered)
    MPI.Waitall(cm.recvrequests)
    MPI.Waitall(cm.sendrequests)

    A[cm.pattern.recvindices] .= cm.recvbufferdevice
end

function cooperative_testall(requests)
    done = false
    while !done
        done = MPI.Testall(requests)
        yield()
    end
end

function cooperative_wait(task::Task)
    while !Base.istaskdone(task)
        MPI.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, MPI.COMM_WORLD)
        yield()
    end
    wait(task)
end

function start!(A, cm::CommManagerTripleBuffered)
    backend = get_backend(cm)

    # We use two host buffers each for send and receive.  We do this to have
    # one buffer pinned by the device stack and one pinned for the network
    # interface.  We see deadlocks if two separate buffers are not used.  See,
    # <https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/> for more
    # details.

    # There is an issue with the non blocking synchronization in CUDA.jl that
    # can cause long pauses:
    #
    #   https://github.com/JuliaGPU/CUDA.jl/issues/1910
    #
    # so we will want to profiling (as done in the issue) to make sure our code
    # is performing as expected.

    # Wait for kernels on the main thread/stream to finish before launching
    # kernels on on different threads which each have their own streams.
    KernelAbstractions.synchronize(backend)

    MPI.Startall(cm.recvrequests)
    cm.recvtask[] = Base.Threads.@spawn begin
        KernelAbstractions.priority!(backend, :high)
        cooperative_testall(cm.recvrequests)
        copyto!(cm.recvbufferhost, cm.recvbuffercomm)
        KernelAbstractions.copyto!(backend, cm.recvbufferdevice, cm.recvbufferhost)
        A[cm.pattern.recvindices] .= cm.recvbufferdevice
        KernelAbstractions.synchronize(backend)
    end

    cm.sendtask[] = Base.Threads.@spawn begin
        cm.sendbufferdevice .= A[cm.pattern.sendindices]
        KernelAbstractions.copyto!(backend, cm.sendbufferhost, cm.sendbufferdevice)
        KernelAbstractions.synchronize(backend)
        copyto!(cm.sendbuffercomm, cm.sendbufferhost)
        MPI.Startall(cm.sendrequests)
        cooperative_testall(cm.sendrequests)
    end
end

function finish!(_, cm::CommManagerTripleBuffered)
    cooperative_wait(cm.recvtask[])
    cooperative_wait(cm.sendtask[])
end

function share!(A, cm::AbstractCommManager)
    start!(A, cm)

    progress(cm)

    finish!(A, cm)
end
