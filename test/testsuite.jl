module Testsuite

using Raven
using Raven.Adapt
using Raven.LinearAlgebra
using Raven.OneDimensionalNodes
import Raven.P4estTypes
using Raven.StaticArrays
using Raven.WriteVTK

using StableRNGs
using Test

include("cells.jl")
include("grids.jl")
include("kron.jl")

function testsuite(AT, FT)
    @testset "Cells ($AT, $FT)" begin
        cells_testsuite(AT, FT)
    end

    # Unfortunately, our KernelAbstractions kernels do not work
    # when FT is not an `isbitstype`.
    if isbitstype(FT)
        @testset "Grid generation ($AT, $FT)" begin
            grids_testsuite(AT, FT)
        end
    end

    @testset "Kronecker operators ($AT, $FT)" begin
        kron_testsuite(AT, FT)
    end
end

end
