using Harpy
using MPI
using Test

MPI.Initialized() || MPI.Init()

Harpy.greet()
