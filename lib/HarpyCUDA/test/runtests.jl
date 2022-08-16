using Harpy
using HarpyCUDA
using CUDA
using MPI
using Test

MPI.Initialized() || MPI.Init()

if CUDA.functional()
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Harpy.Testsuite.testsuite(CuArray, Float32)
else
    error("No CUDA GPUs available!")
end
