using MPI
using Test

MPI.Init()

@test MPI.Comm_size(MPI.COMM_WORLD) == 2
