module RavenCUDAExt

import Raven
import Adapt
import MPI

isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)
isdefined(Base, :get_extension) ? (using CUDA.CUDAKernels) : (using ..CUDA.CUDAKernels)

Raven.get_backend(::Type{T}) where {T<:CuArray} = CUDABackend(; always_inline = true)
Raven.arraytype(::Type{T}) where {T<:CuArray} = CuArray
Raven.arraytype(::Type{T}) where {T<:CuDeviceArray} = CuArray

function Raven.pin(::Type{T}, A::Array) where {T<:CuArray}
    if length(A) > 0
        A = CUDA.Mem.pin(A)
    end

    return A
end

Raven.usetriplebuffer(::Type{T}) where {T<:CuArray} = !MPI.has_cuda()

# Speed up time to first cell by building the cell on the CPU and copying it to
# the device.
function Raven.LobattoCell{Tuple{S1},T,A}() where {S1,T,A<:CuArray}
    cell = Raven.LobattoCell{Tuple{S1},T,Array}()
    return Adapt.adapt(A, cell)
end
function Raven.LobattoCell{Tuple{S1,S2},T,A}() where {S1,S2,T,A<:CuArray}
    cell = Raven.LobattoCell{Tuple{S1,S2},T,Array}()
    return Adapt.adapt(A, cell)
end
function Raven.LobattoCell{Tuple{S1,S2,S3},T,A}() where {S1,S2,S3,T,A<:CuArray}
    cell = Raven.LobattoCell{Tuple{S1,S2,S3},T,Array}()
    return Adapt.adapt(A, cell)
end

function Raven.adaptsparse(::Type{T}, S) where {T<:CuArray}
    return Adapt.adapt(T, Raven.GeneralSparseMatrixCSC(S))
end

function CuArray(a::Raven.GridArray)
    acu = CuArray{eltype(a)}(undef, size(a))
    acu .= a
    return acu
end

Raven.Stream(::CUDABackend) = CuStream()
Raven.synchronize(::CUDABackend, a) = synchronize(a)
Raven.stream!(f::Function, ::CUDABackend, stream::CuStream) = stream!(f, stream)

Adapt.adapt_storage(::CUDA.Adaptor, ::MPI.Comm) = nothing

end # module RavenCUDAExt
