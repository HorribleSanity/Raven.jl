function findorient(::Val{2}, dest::AbstractArray{<:Any,1}, src::AbstractArray{<:Any,1})
    if size(dest) != (2,) && size(src) != (2,)
        throw(ArgumentError("Arguments dest=$dest src=$src need to be of size (2,)"))
    end

    k = dest == src ? 1 : 2

    return Raven.Orientation{2}(k)
end

function findorient(::Val{4}, dest::AbstractArray{<:Any,2}, src::AbstractArray{<:Any,2})
    if size(dest) != (2, 2) && size(src) != (2, 2)
        throw(ArgumentError("Arguments dest=$dest src=$src need to be of size (2,2)"))
    end

    k = 0

    for (i, perm) in enumerate((
        (1, 2, 3, 4),
        (2, 1, 4, 3),
        (3, 4, 1, 2),
        (4, 3, 2, 1),
        (1, 3, 2, 4),
        (2, 4, 1, 3),
        (3, 1, 4, 2),
        (4, 2, 3, 1),
    ))
        if src[perm[1]] == dest[1] &&
           src[perm[2]] == dest[2] &&
           src[perm[3]] == dest[3] &&
           src[perm[4]] == dest[4]
            k = i
            break
        end
    end

    if k == 0
        throw(ArgumentError("Orientation from $src to $dest is unknown"))
    end

    return Raven.Orientation{4}(k)
end

# function findorient(::Val{8}, dest::AbstractArray{<:Any,3}, src::AbstractArray{<:Any,3})
#     if size(dest) != (2, 2, 2) && size(src) != (2, 2, 2)
#         throw(ArgumentError("Arguments dest=$dest src=$src need to be of size (2,2,2)"))
#     end
#
#     k = 0
#
#     for (i, perm) in enumerate((
#         (1, 2, 3, 4, 5, 6, 7, 8),
#         (2, 1, 4, 3, 6, 5, 8, 7),
#         (3, 4, 1, 2, 7, 8, 5, 6),
#         (4, 3, 2, 1, 8, 7, 6, 5),
#         (5, 6, 7, 8, 1, 2, 3, 4),
#         (6, 5, 8, 7, 2, 1, 4, 3),
#         (7, 8, 5, 6, 3, 4, 1, 2),
#         (8, 7, 6, 5, 4, 3, 2, 1),
#         (1, 2, 5, 6, 3, 4, 7, 8),
#         (2, 1, 6, 5, 4, 3, 8, 7),
#         (3, 4, 7, 8, 1, 2, 5, 6),
#         (4, 3, 8, 7, 2, 1, 6, 5),
#         (5, 6, 1, 2, 7, 8, 3, 4),
#         (6, 5, 2, 1, 8, 7, 4, 3),
#         (7, 8, 3, 4, 5, 6, 1, 2),
#         (8, 7, 4, 3, 6, 5, 2, 1),
#         (1, 3, 2, 4, 5, 7, 6, 8),
#         (2, 4, 1, 3, 6, 8, 5, 7),
#         (3, 1, 4, 2, 7, 5, 8, 6),
#         (4, 2, 3, 1, 8, 6, 7, 5),
#         (5, 7, 6, 8, 1, 3, 2, 4),
#         (6, 8, 5, 7, 2, 4, 1, 3),
#         (7, 5, 8, 6, 3, 1, 4, 2),
#         (8, 6, 7, 5, 4, 2, 3, 1),
#         (1, 3, 5, 7, 2, 4, 6, 8),
#         (2, 4, 6, 8, 1, 3, 5, 7),
#         (3, 1, 7, 5, 4, 2, 8, 6),
#         (4, 2, 8, 6, 3, 1, 7, 5),
#         (5, 7, 1, 3, 6, 8, 2, 4),
#         (6, 8, 2, 4, 5, 7, 1, 3),
#         (7, 5, 3, 1, 8, 6, 4, 2),
#         (8, 6, 4, 2, 7, 5, 3, 1),
#         (1, 5, 2, 6, 3, 7, 4, 8),
#         (2, 6, 1, 5, 4, 8, 3, 7),
#         (3, 7, 4, 8, 1, 5, 2, 6),
#         (4, 8, 3, 7, 2, 6, 1, 5),
#         (5, 1, 6, 2, 7, 3, 8, 4),
#         (6, 2, 5, 1, 8, 4, 7, 3),
#         (7, 3, 8, 4, 5, 1, 6, 2),
#         (8, 4, 7, 3, 6, 2, 5, 1),
#         (1, 5, 3, 7, 2, 6, 4, 8),
#         (2, 6, 4, 8, 1, 5, 3, 7),
#         (3, 7, 1, 5, 4, 8, 2, 6),
#         (4, 8, 2, 6, 3, 7, 1, 5),
#         (5, 1, 7, 3, 6, 2, 8, 4),
#         (6, 2, 8, 4, 5, 1, 7, 3),
#         (7, 3, 5, 1, 8, 4, 6, 2),
#         (8, 4, 6, 2, 7, 3, 5, 1),
#     ))
#         if src[perm[1]] == dest[1] &&
#            src[perm[2]] == dest[2] &&
#            src[perm[3]] == dest[3] &&
#            src[perm[4]] == dest[4] &&
#            src[perm[5]] == dest[5] &&
#            src[perm[6]] == dest[6] &&
#            src[perm[7]] == dest[7] &&
#            src[perm[8]] == dest[8]
#             k = i
#             break
#         end
#     end
#
#     if k == 0
#         throw(ArgumentError("Orientation from $src to $dest is unknown"))
#     end
#
#     return Raven.Orientation{8}(k)
# end

@testset "Orientation" begin

    @testset "Orientation{2}" begin
        @test Raven.orientindices(Raven.Orientation{2}(1), (3,)) ==
              [CartesianIndex(1), CartesianIndex(2), CartesianIndex(3)]

        @test Raven.orientindices(Raven.Orientation{2}(2), (3,)) ==
              [CartesianIndex(3), CartesianIndex(2), CartesianIndex(1)]

        L = LinearIndices((1:2,))
        for j = 1:2
            p = Raven.Orientation{2}(j)
            pL = L[Raven.orientindices(p, (2,))]
            for i = 1:2
                q = Raven.Orientation{2}(i)
                qpL = pL[Raven.orientindices(q, (2,))]
                qp = findorient(Val(2), qpL, L)
                @test qp == q ∘ p
            end
        end

        @test inv(Raven.Orientation{2}(1)) == Raven.Orientation{2}(1)
        @test inv(Raven.Orientation{2}(2)) == Raven.Orientation{2}(2)

        L = LinearIndices((1:2,))
        for j = 1:2
            p = Raven.Orientation{2}(j)
            pL = L[Raven.orientindices(p, (2,))]

            q = Raven.orient(Val(2), Tuple(pL))
            @test q == inv(p)
        end

        A = [1, -1]
        for i = 1:2
            o = Raven.Orientation{2}(i)
            p = findorient(Val(2), A[Raven.orientindices(o, size(A))], A)
            @test o == p
        end

        @test Raven.perm(Raven.Orientation{2}(1)) == SA[1, 2]
        @test Raven.perm(Raven.Orientation{2}(2)) == SA[2, 1]
    end

    @testset "Orientation{4}" begin
        dims = (3, 4)
        A = reshape(1:prod(dims), dims)

        L = LinearIndices((1:2, 1:2))
        for j = 1:8
            p = Raven.Orientation{4}(j)
            pL = L[Raven.orientindices(p, (2, 2))]
            for i = 1:8
                q = Raven.Orientation{4}(i)
                qpL = pL[Raven.orientindices(q, (2, 2))]
                qp = findorient(Val(4), qpL, L)
                @test qp == q ∘ p
            end
        end

        @test inv(Raven.Orientation{4}(1)) == Raven.Orientation{4}(1)
        @test inv(Raven.Orientation{4}(2)) == Raven.Orientation{4}(2)
        @test inv(Raven.Orientation{4}(3)) == Raven.Orientation{4}(3)
        @test inv(Raven.Orientation{4}(4)) == Raven.Orientation{4}(4)
        @test inv(Raven.Orientation{4}(5)) == Raven.Orientation{4}(5)
        @test inv(Raven.Orientation{4}(6)) == Raven.Orientation{4}(7)
        @test inv(Raven.Orientation{4}(7)) == Raven.Orientation{4}(6)
        @test inv(Raven.Orientation{4}(8)) == Raven.Orientation{4}(8)

        L = LinearIndices((1:2, 1:2))
        for j = 1:8
            p = Raven.Orientation{4}(j)
            pL = L[Raven.orientindices(p, (2, 2))]

            q = Raven.orient(Val(4), Tuple(pL))
            @test q == inv(p)
        end

        A = reshape([1, 30, 23, -1], (2, 2))
        for i = 1:8
            o = Raven.Orientation{4}(i)
            p = findorient(Val(4), A[Raven.orientindices(o, size(A))], A)
            @test o == p
        end

        for (o, perm) in enumerate((
            SA[1, 2, 3, 4],
            SA[2, 1, 4, 3],
            SA[3, 4, 1, 2],
            SA[4, 3, 2, 1],
            SA[1, 3, 2, 4],
            SA[2, 4, 1, 3],
            SA[3, 1, 4, 2],
            SA[4, 2, 3, 1],
        ))
            src = reshape(1:4, (2, 2))
            dest = reshape(perm, (2, 2))
            @test findorient(Val(4), dest, src) == Raven.Orientation{4}(o)
            @test perm == Raven.perm(Raven.Orientation{4}(o))
        end
    end

    # @testset "Orientation{8}" begin
    #     dims = (3, 4, 5)
    #     A = reshape(1:prod(dims), dims)
    #
    #     for i = 1:48
    #         o = Raven.Orientation{8}(i)
    #         B = A[Raven.orientindices(o, dims)]
    #         C = B[Raven.orientindices(o, dims, true)]
    #         @test A == C
    #     end
    #
    #     A = reshape([1, 30, 23, 32, 4, 9, 99, -1], (2, 2, 2))
    #     for i = 1:48
    #         o = Raven.Orientation{8}(i)
    #         p = findorient(Val(8), A[Raven.orientindices(o, size(A))], A)
    #         @test o == p
    #     end
    #
    #     for (o, perm) in enumerate((
    #         SA[1, 2, 3, 4, 5, 6, 7, 8],
    #         SA[2, 1, 4, 3, 6, 5, 8, 7],
    #         SA[3, 4, 1, 2, 7, 8, 5, 6],
    #         SA[4, 3, 2, 1, 8, 7, 6, 5],
    #         SA[5, 6, 7, 8, 1, 2, 3, 4],
    #         SA[6, 5, 8, 7, 2, 1, 4, 3],
    #         SA[7, 8, 5, 6, 3, 4, 1, 2],
    #         SA[8, 7, 6, 5, 4, 3, 2, 1],
    #         SA[1, 2, 5, 6, 3, 4, 7, 8],
    #         SA[2, 1, 6, 5, 4, 3, 8, 7],
    #         SA[3, 4, 7, 8, 1, 2, 5, 6],
    #         SA[4, 3, 8, 7, 2, 1, 6, 5],
    #         SA[5, 6, 1, 2, 7, 8, 3, 4],
    #         SA[6, 5, 2, 1, 8, 7, 4, 3],
    #         SA[7, 8, 3, 4, 5, 6, 1, 2],
    #         SA[8, 7, 4, 3, 6, 5, 2, 1],
    #         SA[1, 3, 2, 4, 5, 7, 6, 8],
    #         SA[2, 4, 1, 3, 6, 8, 5, 7],
    #         SA[3, 1, 4, 2, 7, 5, 8, 6],
    #         SA[4, 2, 3, 1, 8, 6, 7, 5],
    #         SA[5, 7, 6, 8, 1, 3, 2, 4],
    #         SA[6, 8, 5, 7, 2, 4, 1, 3],
    #         SA[7, 5, 8, 6, 3, 1, 4, 2],
    #         SA[8, 6, 7, 5, 4, 2, 3, 1],
    #         SA[1, 3, 5, 7, 2, 4, 6, 8],
    #         SA[2, 4, 6, 8, 1, 3, 5, 7],
    #         SA[3, 1, 7, 5, 4, 2, 8, 6],
    #         SA[4, 2, 8, 6, 3, 1, 7, 5],
    #         SA[5, 7, 1, 3, 6, 8, 2, 4],
    #         SA[6, 8, 2, 4, 5, 7, 1, 3],
    #         SA[7, 5, 3, 1, 8, 6, 4, 2],
    #         SA[8, 6, 4, 2, 7, 5, 3, 1],
    #         SA[1, 5, 2, 6, 3, 7, 4, 8],
    #         SA[2, 6, 1, 5, 4, 8, 3, 7],
    #         SA[3, 7, 4, 8, 1, 5, 2, 6],
    #         SA[4, 8, 3, 7, 2, 6, 1, 5],
    #         SA[5, 1, 6, 2, 7, 3, 8, 4],
    #         SA[6, 2, 5, 1, 8, 4, 7, 3],
    #         SA[7, 3, 8, 4, 5, 1, 6, 2],
    #         SA[8, 4, 7, 3, 6, 2, 5, 1],
    #         SA[1, 5, 3, 7, 2, 6, 4, 8],
    #         SA[2, 6, 4, 8, 1, 5, 3, 7],
    #         SA[3, 7, 1, 5, 4, 8, 2, 6],
    #         SA[4, 8, 2, 6, 3, 7, 1, 5],
    #         SA[5, 1, 7, 3, 6, 2, 8, 4],
    #         SA[6, 2, 8, 4, 5, 1, 7, 3],
    #         SA[7, 3, 5, 1, 8, 4, 6, 2],
    #         SA[8, 4, 6, 2, 7, 3, 5, 1],
    #     ))
    #         src = reshape(1:8, (2, 2, 2))
    #         dest = reshape(perm, (2, 2, 2))
    #         @test findorient(Val(8), dest, src) == Raven.Orientation{8}(o)
    #     end
    # end
end
