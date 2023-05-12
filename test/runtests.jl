using CUDA
using CUDA.CUDAKernels
using MPI
using Pkg
using Raven
import Raven.P4estTypes
using Raven.StaticArrays
using Test

function runmpitests()
    test_dir = @__DIR__
    istest(f) = endswith(f, ".jl") && startswith(f, "mpitest_")
    testfiles = sort(filter(istest, readdir(test_dir)))

    mktempdir() do tmp_dir
        base_dir = joinpath(@__DIR__, "..")

        # Change to temporary directory so that any files created by the
        # example get cleaned up after execution.
        cd(tmp_dir)
        test_project = Pkg.Types.projectfile_path(test_dir)
        tmp_project = Pkg.Types.projectfile_path(tmp_dir)
        cp(test_project, tmp_project)

        # Copy data files to temporary directory
        # test_data_dir = joinpath(test_dir, "data")
        # tmp_data_dir = joinpath(tmp_dir, "data")
        # mkdir(tmp_data_dir)
        # for f in readdir(test_data_dir)
        #     cp(joinpath(test_data_dir, f), joinpath(tmp_data_dir, f))
        # end

        # Setup MPI and P4est preferences
        code = "import Pkg; Pkg.develop(path=raw\"$base_dir\"); Pkg.instantiate(); Pkg.precompile(); include(joinpath(raw\"$test_dir\", \"configure_packages.jl\"))"
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$tmp_project -e "$code"`
        @info "Initializing MPI and P4est with" cmd
        @test success(pipeline(cmd, stderr = stderr, stdout = stdout))

        @info "Running MPI tests..."
        @testset "$f" for f in testfiles
            nprocs = parse(Int, first(match(r"_n(\d*)_", f).captures))
            cmd = `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --startup-file=no --project=$tmp_project $(joinpath(test_dir, f))`
            @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
        end
    end
end

MPI.Initialized() || MPI.Init()

include("arrays.jl")
include("communication.jl")
include("gridnumbering.jl")

include("testsuite.jl")

Testsuite.testsuite(Array, Float64)
Testsuite.testsuite(Array, BigFloat)

if CUDA.functional()
    @info "Running test suite with CUDA"
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Testsuite.testsuite(CuArray, Float32)
end

runmpitests()
