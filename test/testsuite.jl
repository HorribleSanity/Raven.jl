module Testsuite

using ..Harpy
using ..Harpy.Adapt
using ..Harpy.FillArrays
using ..Harpy.LinearAlgebra
using ..Harpy.OneDimensionalNodes
using ..Harpy.StaticArrays

using StableRNGs
using Test

include("cells.jl")
include("kron.jl")

function testsuite(AT, FT)
    @testset "Cells ($AT, $FT)" begin
        cells_testsuite(AT, FT)
    end

    @testset "Kronecker operators ($AT, $FT)" begin
        kron_testsuite(AT, FT)
    end
end

end
