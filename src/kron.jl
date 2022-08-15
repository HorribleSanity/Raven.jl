struct Kron{T}
    args::T
    Kron(args::Tuple) = new{typeof(args)}(args)
end

Adapt.adapt_structure(to, K::Kron) = Kron(map(x -> Adapt.adapt(to, x), K.args))
components(K::Kron) = K.args
Base.collect(K::Kron{Tuple{T}}) where {T} = collect(K.args[1])
Base.collect(K::Kron) = collect(kron(K.args...))
Base.size(K::Kron, j::Int) = prod(size.(K.args, j))

import Base.==
==(J::Kron, K::Kron) = all(J.args .== K.args)

import Base.*

@kernel function kron_D_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[i, l], g[l, j], acc)
        end
        r[i, j] = acc
    end
end

function (*)(K::Kron{Tuple{D}}, f::F) where {D<:AbstractMatrix,F<:AbstractVecOrMat}
    (d,) = components(K)

    g = reshape(f, size(d, 2), :)
    r = similar(f, size(d, 1), size(g, 2))

    device = get_device(r)
    kernel! = kron_D_kernel(device, size(r, 1))
    event = Event(device)
    event = kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r), dependencies = event)
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_ED_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j, k) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[i, l], g[l, j, k], acc)
        end
        r[i, j, k] = acc
    end
end

function (*)(K::Kron{Tuple{E,D}}, f::F) where {D<:AbstractMatrix,E<:Eye,F<:AbstractVecOrMat}
    e, d = components(K)

    g = reshape(f, size(d, 2), size(e, 1), :)
    r = similar(f, size(d, 1), size(e, 1), size(g, 3))

    device = get_device(r)
    kernel! = kron_ED_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r), dependencies = event)
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_DE_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j, k) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[j, l], g[i, l, k], acc)
        end
        r[i, j, k] = acc
    end
end

function (*)(K::Kron{Tuple{D,E}}, f::F) where {D<:AbstractMatrix,E<:Eye,F<:AbstractVecOrMat}
    d, e = components(K)

    g = reshape(f, size(e, 1), size(d, 2), :)
    r = similar(f, size(e, 1), size(d, 1), size(g, 3))

    device = get_device(r)
    kernel! = kron_DE_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r), dependencies = event)
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_BA_kernel(
    r::AbstractArray{T},
    a,
    b,
    g,
    ::Val{L},
    ::Val{M},
) where {T,L,M}
    (i, j, k) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        # TODO: use temporary space to reduce complexity
        @unroll for m in M
            @unroll for l in L
                acc = muladd(b[j, m] * a[i, l], g[l, m, k], acc)
            end
        end
        r[i, j, k] = acc
    end
end

function (*)(
    K::Kron{Tuple{B,A}},
    f::F,
) where {A<:AbstractMatrix,B<:AbstractMatrix,F<:AbstractVecOrMat}
    b, a = components(K)

    g = reshape(f, size(a, 2), size(b, 2), :)
    r = similar(f, size(a, 1), size(b, 1), size(g, 3))

    device = get_device(r)
    kernel! = kron_BA_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(
        r,
        a,
        b,
        g,
        Val(axes(a, 2)),
        Val(axes(b, 2));
        ndrange = size(r),
        dependencies = event,
    )
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_EED_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j, k, e) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[i, l], g[l, j, k, e], acc)
        end
        r[i, j, k, e] = acc
    end
end

function (*)(
    K::Kron{Tuple{E₃,E₂,D}},
    f::F,
) where {D<:AbstractMatrix,E₃<:Eye,E₂<:Eye,F<:AbstractVecOrMat}
    e₃, e₂, d = components(K)

    g = reshape(f, size(d, 2), size(e₂, 1), size(e₃, 1), :)
    r = similar(f, size(d, 1), size(e₂, 1), size(e₃, 1), size(g, 4))

    device = get_device(r)
    kernel! = kron_EED_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r), dependencies = event)
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_EDE_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j, k, e) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[j, l], g[i, l, k, e], acc)
        end
        r[i, j, k, e] = acc
    end
end

function (*)(
    K::Kron{Tuple{E₃,D,E₁}},
    f::F,
) where {D<:AbstractMatrix,E₁<:Eye,E₃<:Eye,F<:AbstractVecOrMat}
    e₃, d, e₁ = components(K)

    g = reshape(f, size(e₁, 1), size(d, 2), size(e₃, 1), :)
    r = similar(f, size(e₁, 1), size(d, 1), size(e₃, 1), size(g, 4))

    device = get_device(r)
    kernel! = kron_EDE_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r), dependencies = event)
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_DEE_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j, k, e) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[k, l], g[i, j, l, e], acc)
        end
        r[i, j, k, e] = acc
    end
end

function (*)(
    K::Kron{Tuple{D,E₂,E₁}},
    f::F,
) where {D<:AbstractMatrix,E₁<:Eye,E₂<:Eye,F<:AbstractVecOrMat}
    d, e₂, e₁ = components(K)

    g = reshape(f, size(e₁, 1), size(e₂, 1), size(d, 2), :)
    r = similar(f, size(e₁, 1), size(e₂, 1), size(d, 1), size(g, 4))

    device = get_device(r)
    kernel! = kron_DEE_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r), dependencies = event)
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

@kernel function kron_CBA_kernel(
    r::AbstractArray{T},
    a,
    b,
    c,
    g,
    ::Val{L},
    ::Val{M},
    ::Val{N},
) where {T,L,M,N}
    (i, j, k, e) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        # TODO: use temporary space to reduce complexity
        @unroll for l in L
            @unroll for m in M
                @unroll for n in N
                    acc = muladd(c[k, n] * b[j, m] * a[i, l], g[l, m, n, e], acc)
                end
            end
        end
        r[i, j, k, e] = acc
    end
end

function (*)(
    K::Kron{Tuple{C,B,A}},
    f::F,
) where {A<:AbstractMatrix,B<:AbstractMatrix,C<:AbstractMatrix,F<:AbstractVecOrMat}
    c, b, a = components(K)

    g = reshape(f, size(a, 2), size(b, 2), size(c, 2), :)
    r = similar(f, size(a, 1), size(b, 1), size(c, 1), size(g, 4))

    device = get_device(r)
    kernel! = kron_CBA_kernel(device, size(r)[begin:end-1])
    event = Event(device)
    event = kernel!(
        r,
        a,
        b,
        c,
        g,
        Val(axes(a, 2)),
        Val(axes(b, 2)),
        Val(axes(c, 2));
        ndrange = size(r),
        dependencies = event,
    )
    wait(device, event)

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end
