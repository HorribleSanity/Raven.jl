module Raven

using Adapt
using FillArrays
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using MPI
using OneDimensionalNodes
import P4estTypes
using RecipesBase
using PrecompileTools
using StaticArrays
using StaticArrays: tuple_prod, tuple_length, size_to_tuple
using WriteVTK

export LobattoCell

export arraytype, floattype
export derivatives, facemass, mass, points, toequallyspaced
export derivatives_1d, points_1d, weights_1d
export referencecell, levels, trees, offset

export brick

export GridManager, generate

export adapt!

include("arrays.jl")
include("kron.jl")
include("cells.jl")
include("coarsegrids.jl")
include("grids.jl")
include("gridmanager.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/CUDAExt.jl")
    end
end

@setup_workload begin
    @compile_workload begin
        include("../test/testsuite.jl")
        for T in (Float64, Float32, BigFloat)
            Testsuite.cells_testsuite(Array, T)
            Testsuite.kron_testsuite(Array, T)
        end
    end
end

end # module Raven
