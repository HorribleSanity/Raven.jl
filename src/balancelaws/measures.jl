export weightednorm
export entropyintegral

using LinearAlgebra: norm, dot

weightednorm(dg, q, p = 2; componentwise = false) =
    _weightednorm(q, Val(p), dg.MJ, dg.comm, componentwise)

function _weightednorm(q, ::Val{p}, MJ, comm, componentwise) where {p}
    n = map(components(q)) do f
        MPI.Allreduce(sum(MJ .* abs.(f) .^ p), MPI.SUM, comm)^(1 // p)
    end
    componentwise ? n : norm(n, p)
end

function _weightednorm(q, ::Val{Inf}, _, comm, componentwise)
    n = map(components(q)) do f
        MPI.Allreduce(maximum(abs.(f)), MPI.MAX, comm)
    end
    componentwise ? n : norm(n, Inf)
end

function entropyintegral(dg, q)
    η = entropy.(Ref(dg.law), q, dg.auxstate)
    MPI.Allreduce(sum(dg.MJ .* η), MPI.SUM, dg.comm)
    #sum(dg.MJ .* entropy.(Ref(dg.law), q, dg.auxstate))
end
function entropyproduct(dg, p, q)
    v = entropyvariables.(Ref(dg.law), p, dg.auxstate)
    vᵀq = dot.(v, q)
    MPI.Allreduce(sum(dg.MJ .* vᵀq), MPI.SUM, dg.comm)
    #sum(dg.MJ .* dot.(entropyvariables.(Ref(dg.law), p, dg.auxstate), q))
end
