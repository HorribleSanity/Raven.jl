module Raven

import AcceleratedKernels as AK
using Adapt
using Compat
using GPUArraysCore
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

export LobattoCell, GaussCell

export arraytype, floattype
export derivatives, facemass, mass, points, toequallyspaced
export togauss, tolobatto, toboundary
export weightedderivatives, skewweightedderivatives
export derivatives_1d, points_1d, weights_1d, tohalves_1d
export togauss_1d, tolobatto_1d, toboundary_1d
export weightedderivatives_1d, skewweightedderivatives_1d
export referencecell, levels, trees, offset, numcells, facecodes
export continuoustodiscontinuous, nodecommpattern
export communicatingcells, noncommunicatingcells
export facemaps, boundarycodes
export decode
export commmanager, share!, start!, finish!, progress
export faceoffsets, facedims, faceviews

export volumemetrics, surfacemetrics
export min_node_distance

export flatten, unflatten

export brick, coarsegrid, extrude

export GridManager, generate
export GridArray, components, sizewithghosts, viewwithghosts, parentwithghosts

export adapt!

include("orientation.jl")
include("fileio.jl")
include("arrays.jl")
include("sparsearrays.jl")
include("streams.jl")
include("eye.jl")
include("facecode.jl")
include("flatten.jl")
include("cells.jl")
include("lobattocells.jl")
include("gausscells.jl")
include("coarsegrids.jl")
include("communication.jl")
include("grids.jl")
include("gridmanager.jl")
include("gridarrays.jl")
include("kron.jl")

include("balancelaws/BalanceLaws.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include(
            "../ext/RavenCUDAExt.jl",
        )
        @require WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192" include(
            "../ext/RavenWriteVTKExt.jl",
        )
    end

    MPI.add_finalize_hook!() do
        for cm in COMM_MANAGERS
            finalize(cm.value)
        end
    end
end

end # module Raven
