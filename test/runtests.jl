using CUDA
using CUDA.CUDAKernels
using Harpy
using MPI
using Test

MPI.Initialized() || MPI.Init()

Harpy.Testsuite.testsuite(Array, Float64)
Harpy.Testsuite.testsuite(Array, BigFloat)

if CUDA.functional()
    @info "Running test suite with CUDA"
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Harpy.Testsuite.testsuite(CuArray, Float32)
end
