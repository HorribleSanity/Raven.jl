export weightednorm
export entropyintegral

using LinearAlgebra: norm, dot

weightednorm(dg, q, p = 2; componentwise = false) =
    weightednorm(q, Val(p), dg.MJ, componentwise)

function weightednorm(q, ::Val{p}, MJ, componentwise) where {p}
    n = map(components(q)) do f
        sum(MJ .* abs.(f) .^ p) ^ (1 // p)
    end
    componentwise ? n : norm(n, p)
end

function weightednorm(q, ::Val{Inf}, _, componentwise)
    n = map(components(q)) do f
        maximum(abs.(f))
    end
    componentwise ? n : norm(n, Inf)
end

function entropyintegral(dg, q)
    η = entropy.(Ref(dg.law), q, dg.auxstate)
    sum(dg.MJ .* η)
    #sum(dg.MJ .* entropy.(Ref(dg.law), q, dg.auxstate))
end
function entropyproduct(dg, p, q)
    v = entropyvariables.(Ref(dg.law), p, dg.auxstate)
    vᵀq = dot.(v, q)
    sum(dg.MJ .* vᵀq)
    #sum(dg.MJ .* dot.(entropyvariables.(Ref(dg.law), p, dg.auxstate), q))
end
