using Harpy
using MPI
using Test

MPI.Initialized() || MPI.Init()

Harpy.Testsuite.testsuite(Array, Float64)
Harpy.Testsuite.testsuite(Array, BigFloat)
