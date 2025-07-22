function Bennu.batchedbandedmatrix(dg::DGSEM, q, dq, element_bandwidth = 1; time = 0)
    function matvec!(dq, q, event)
        wait(event)
        dg(dq, q, time; increment = false)
        return Event(getdevice(dg))
    end
    threads = 256
    mat = Bennu.batchedbandedmatrix(matvec!, dg.grid, dq, q, element_bandwidth, threads)
end

function batchedbandedlu(dg::DGSEM, q, dq, element_bandwidth = 1; time = 0)
    mat = Bennu.batchedbandedmatrix(dg, dq, q, element_bandwidth)
    return batchedbandedlu!(mat.data)
end
