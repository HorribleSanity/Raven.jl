function scatter!(dest, src, grid::Grid)
    dtoc = discontinuoustocontinuous(grid)
    cm = commmanager(eltype(src), cnodecommpattern(grid))
    scatter!(dest, src, dtoc, cm)
end

function scatter!(dest, src, dtoc, cm::AbstractCommManager)
    scatterstart!(dest, src, dtoc, cm)
    scatterfinish!(dest, src, dtoc, cm)
end

function scatterstart!(_, src, _, cm)
    src = viewwithghosts(src)
    start!(src, cm)
end

@kernel function scatter_kernel!(dest, src, dtoc)
    i = @index(Global)
    @inbounds dest[i] = src[dtoc[i]]
end

function scatterfinish!(dest, src, dtoc, cm)
    dest = viewwithghosts(dest)
    src = viewwithghosts(src)

    finish!(src, cm)

    axes(dest) == axes(dtoc) || Broadcast.throwdm(axes(dest), axes(dtoc))
    isempty(dest) && return

    setbuffer_kernel!(get_backend(dest), 256)(dest, src, dtoc; ndrange = length(dest))

    return
end

function gather!(dest, src, grid::Grid; op = +, init = zero(eltype(src)))
    ctod = continuoustodiscontinuous(grid)
    cm = commmanager(eltype(src), nodecommpattern(grid))
    gather!(dest, src, ctod, cm; op, init)
end

function gather!(dest, src, ctod, cm::AbstractCommManager; op = +, init = zero(eltype(src)))
    gatherstart!(dest, src, ctod, cm; op, init)
    gatherfinish!(dest, src, ctod, cm; op, init)
end

function gatherstart!(
    dest,
    src,
    ctod,
    cm::AbstractCommManager;
    op = +,
    init = zero(eltype(src)),
)
    src = viewwithghosts(src)
    start!(src, cm)
end

@kernel function gather_kernel!(dest, src, ctod, op, init)
    j = @index(Global)

    @inbounds begin
        val = init
        rows = rowvals(ctod)
        for i in nzrange(ctod, j)
            row = rows[i]
            val = op(val, src[row])
        end
        dest[j] = val
    end
end

function gatherfinish!(
    dest,
    src,
    ctod,
    cm::AbstractCommManager;
    op = +,
    init = zero(eltype(src)),
)
    src = viewwithghosts(src)

    finish!(src, cm)

    (m, n) = size(ctod)

    if length(dest) > n || m > prod(size(src))
        throw(DimensionMismatch("dest and/or src are incompatible with ctod"))
    end

    isempty(dest) && return

    gather_kernel!(get_backend(dest), 256)(
        dest,
        src,
        ctod,
        op,
        init;
        ndrange = length(dest),
    )

    return
end
