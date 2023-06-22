function flatten_testsuite(AT, FT)
    obj = (a = (Complex(FT(1), FT(2)), FT(3)), b = FT(4), c = SVector(FT(5)))
    obj_flattened = (FT(1), FT(2), FT(3), FT(4), FT(5))
    @test @inferred(flatten(obj)) == obj_flattened
    @test @inferred(unflatten(typeof(obj), obj_flattened)) == obj

    @test @inferred(flatten(zero(FT))) == (zero(FT),)
    @test @inferred(unflatten(FT, (zero(FT),))) == zero(FT)

    @kernel function foo!(y, @Const(x), ::Type{T}) where {T}
        @inbounds begin
            z = (x[1], x[2], x[3])
            y[1] = unflatten(T, z)
        end
    end

    @kernel function bar!(x, @Const(y))
        @inbounds begin
            z = flatten(y[1])
            (x[1], x[2], x[3]) = z
        end
    end

    T = Tuple{SVector{1,Complex{FT}},FT}
    xh = [FT(1), FT(2), FT(3)]
    x = AT(xh)
    y = AT{T}([(SVector(complex(zero(FT), zero(FT))), zero(FT))])

    backend = get_backend(x)

    foo!(backend, 1, 1)(y, x, T)
    KernelAbstractions.synchronize(backend)
    @test Array(y)[1] == unflatten(T, xh)

    fill!(x, 0.0)
    bar!(backend, 1, 1)(x, y)
    KernelAbstractions.synchronize(backend)
    @test Array(x) == xh

    return
end
