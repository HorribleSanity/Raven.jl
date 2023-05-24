module Raven

using Adapt
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using MPI
using OneDimensionalNodes
import P4estTypes
using RecipesBase
using StaticArrays
using StaticArrays: tuple_prod, tuple_length, size_to_tuple
using SparseArrays
using WriteVTK

export LobattoCell

export arraytype, floattype
export derivatives, facemass, mass, points, toequallyspaced
export derivatives_1d, points_1d, weights_1d
export referencecell, levels, trees, offset, numcells

export brick, coarsegrid

export GridManager, generate

export adapt!

include("arrays.jl")
include("eye.jl")
include("kron.jl")
include("cells.jl")
include("coarsegrids.jl")
include("communication.jl")
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

end # module Raven
