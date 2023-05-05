using CUDA
using CUDA.CUDAKernels
using Raven
using MPI
using Test

MPI.Initialized() || MPI.Init()

Raven.Testsuite.testsuite(Array, Float64)
Raven.Testsuite.testsuite(Array, BigFloat)

if CUDA.functional()
    @info "Running test suite with CUDA"
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Raven.Testsuite.testsuite(CuArray, Float32)
end
