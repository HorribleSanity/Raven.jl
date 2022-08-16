module HarpyCUDA

using Adapt
using CUDAKernels
using CUDA
using Harpy

Harpy.get_device(::Type{T}) where {T<:CuArray} = CUDADevice()
Harpy.arraytype(::Type{T}) where {T<:CuArray} = CuArray

# Speed up time to first cell by building the cell on the CPU and copying it to
# the device.
function Harpy.LobattoCell{T,A}(dims...) where {T,A<:CuArray}
    cell = LobattoCell{T,Array}(dims...)
    return adapt(A, cell)
end

end # module HarpyCUDA
