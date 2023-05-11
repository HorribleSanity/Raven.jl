module CUDAExt

import Raven
import Adapt
import MPI
using PrecompileTools

isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)
isdefined(Base, :get_extension) ? (using CUDA.CUDAKernels) : (using ..CUDA.CUDAKernels)

Raven.get_backend(::Type{T}) where {T<:CuArray} = CUDABackend(; always_inline = true)
Raven.arraytype(::Type{T}) where {T<:CuArray} = CuArray

Raven.pin(::Type{T}, A::Array) where {T<:CuArray} = CUDA.Mem.pin(A)

Raven.usetriplebuffer(::Type{T}) where {T<:CuArray} = !MPI.has_cuda()

# Speed up time to first cell by building the cell on the CPU and copying it to
# the device.
function Raven.LobattoCell{T,A}(dims...) where {T,A<:CuArray}
    cell = Raven.LobattoCell{T,Array}(dims...)
    return Adapt.adapt(A, cell)
end

@setup_workload begin
    @compile_workload begin
        for T in (Float64, Float32)
            Raven.Testsuite.cells_testsuite(CuArray, T)
            Raven.Testsuite.kron_testsuite(CuArray, T)
        end
    end
end

end # module CUDAExt
