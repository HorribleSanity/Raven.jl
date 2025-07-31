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

    backend = get_backend(r)
    kernel! = kron_D_kernel(backend, size(r, 1))
    kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{D}}, f::GridArray) where {D<:AbstractMatrix}
    (d,) = components(K)

    r = similar(f, size(d, 1), size(f, 2))

    backend = get_backend(r)
    kernel! = kron_D_kernel(backend, size(r, 1))
    kernel!(r, d, f, Val(axes(d, 2)); ndrange = size(r))

    return r
end

@kernel function kron_E_D_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
    (i, j, k) = @index(Global, NTuple)
    acc = zero(T)
    @inbounds begin
        @unroll for l in L
            acc = muladd(d[i, l], g[l, j, k], acc)
        end
        r[i, j, k] = acc
    end
end

@inline (*)(::Kron{Tuple{E₂,E₁}}, f::F) where {E₁<:Eye,E₂<:Eye,F<:AbstractVecOrMat} =
    copy(f)
@inline (*)(
    ::Kron{Tuple{E₃,E₂,E₁}},
    f::F,
) where {E₁<:Eye,E₂<:Eye,E₃<:Eye,F<:AbstractVecOrMat} = copy(f)
@inline (*)(::Kron{Tuple{E₂,E₁}}, f::F) where {E₁<:Eye,E₂<:Eye,F<:GridVecOrMat} = copy(f)
@inline (*)(::Kron{Tuple{E₃,E₂,E₁}}, f::F) where {E₁<:Eye,E₂<:Eye,E₃<:Eye,F<:GridVecOrMat} =
    copy(f)
@inline (*)(::Kron{Tuple{E₂,E₁}}, f::F) where {E₁<:Eye,E₂<:Eye,F<:GridArray} = copy(f)
@inline (*)(::Kron{Tuple{E₃,E₂,E₁}}, f::F) where {E₁<:Eye,E₂<:Eye,E₃<:Eye,F<:GridArray} =
    copy(f)

function (*)(K::Kron{Tuple{E,D}}, f::F) where {D<:AbstractMatrix,E<:Eye,F<:AbstractVecOrMat}
    e, d = components(K)

    g = reshape(f, size(d, 2), size(e, 1), :)
    r = similar(f, size(d, 1), size(e, 1), size(g, 3))

    backend = get_backend(r)
    kernel! = kron_E_D_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{E,D}}, f::GridArray) where {D<:AbstractMatrix,E<:Eye}
    e, d = components(K)

    r = similar(f, size(d, 1), size(e, 1), size(f, 3))

    backend = get_backend(r)
    kernel! = kron_E_D_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, f, Val(axes(d, 2)); ndrange = size(r))

    return r
end

@kernel function kron_D_E_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
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

    backend = get_backend(r)
    kernel! = kron_D_E_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{D,E}}, f::GridArray) where {D<:AbstractMatrix,E<:Eye}
    d, e = components(K)

    r = similar(f, size(e, 1), size(d, 1), size(f, 3))

    backend = get_backend(r)
    kernel! = kron_D_E_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, f, Val(axes(d, 2)); ndrange = size(r))

    return r
end

@kernel function kron_B_A_kernel(
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

    backend = get_backend(r)
    kernel! = kron_B_A_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, a, b, g, Val(axes(a, 2)), Val(axes(b, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{B,A}}, f::GridArray) where {A<:AbstractMatrix,B<:AbstractMatrix}
    b, a = components(K)

    r = similar(f, size(a, 1), size(b, 1), size(f, 3))

    backend = get_backend(r)
    kernel! = kron_B_A_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, a, b, f, Val(axes(a, 2)), Val(axes(b, 2)); ndrange = size(r))

    return r
end

@kernel function kron_E_E_D_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
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

    backend = get_backend(r)
    kernel! = kron_E_E_D_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(
    K::Kron{Tuple{E₃,E₂,D}},
    f::GridArray,
) where {D<:AbstractMatrix,E₃<:Eye,E₂<:Eye}
    e₃, e₂, d = components(K)

    r = similar(f, size(d, 1), size(e₂, 1), size(e₃, 1), size(f, 4))

    backend = get_backend(r)
    kernel! = kron_E_E_D_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, f, Val(axes(d, 2)); ndrange = size(r))

    return r
end

@kernel function kron_E_D_E_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
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

    backend = get_backend(r)
    kernel! = kron_E_D_E_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(
    K::Kron{Tuple{E₃,D,E₁}},
    f::GridArray,
) where {D<:AbstractMatrix,E₁<:Eye,E₃<:Eye}
    e₃, d, e₁ = components(K)

    r = similar(f, size(e₁, 1), size(d, 1), size(e₃, 1), size(f, 4))

    backend = get_backend(r)
    kernel! = kron_E_D_E_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, f, Val(axes(d, 2)); ndrange = size(r))

    return r
end

@kernel function kron_D_E_E_kernel(r::AbstractArray{T}, d, g, ::Val{L}) where {T,L}
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

    backend = get_backend(r)
    kernel! = kron_D_E_E_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, g, Val(axes(d, 2)); ndrange = size(r))

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(
    K::Kron{Tuple{D,E₂,E₁}},
    f::GridArray,
) where {D<:AbstractMatrix,E₁<:Eye,E₂<:Eye}
    d, e₂, e₁ = components(K)

    r = similar(f, size(e₁, 1), size(e₂, 1), size(d, 1), size(f, 4))

    backend = get_backend(r)
    kernel! = kron_D_E_E_kernel(backend, size(r)[begin:(end-1)])
    kernel!(r, d, f, Val(axes(d, 2)); ndrange = size(r))

    return r
end

@kernel function kron_C_B_A_kernel(
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

    backend = get_backend(r)
    kernel! = kron_C_B_A_kernel(backend, size(r)[begin:(end-1)])
    kernel!(
        r,
        a,
        b,
        c,
        g,
        Val(axes(a, 2)),
        Val(axes(b, 2)),
        Val(axes(c, 2));
        ndrange = size(r),
    )

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(
    K::Kron{Tuple{C,B,A}},
    f::GridArray,
) where {A<:AbstractMatrix,B<:AbstractMatrix,C<:AbstractMatrix}
    c, b, a = components(K)

    r = similar(f, size(a, 1), size(b, 1), size(c, 1), size(f, 4))

    backend = get_backend(r)
    kernel! = kron_C_B_A_kernel(backend, size(r)[begin:(end-1)])
    kernel!(
        r,
        a,
        b,
        c,
        f,
        Val(axes(a, 2)),
        Val(axes(b, 2)),
        Val(axes(c, 2));
        ndrange = size(r),
    )

    return r
end
