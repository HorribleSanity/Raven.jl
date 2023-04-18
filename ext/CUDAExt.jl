module CUDAExt

import Harpy
import Adapt
isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)
isdefined(Base, :get_extension) ? (using CUDA.CUDAKernels) : (using ..CUDA.CUDAKernels)

Harpy.get_device(::Type{T}) where {T<:CuArray} = CUDABackend()
Harpy.arraytype(::Type{T}) where {T<:CuArray} = CuArray

Harpy.pin(::Type{T}, A::Array) where {T<:CuArray} = CUDA.Mem.pin(A)

# Speed up time to first cell by building the cell on the CPU and copying it to
# the device.
function Harpy.LobattoCell{T,A}(dims...) where {T,A<:CuArray}
    cell = Harpy.LobattoCell{T,Array}(dims...)
    return Adapt.adapt(A, cell)
end

end # module CUDAExt
