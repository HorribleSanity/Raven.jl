using CUDA
using CUDA.CUDAKernels
using MPI
using Pkg
using Raven
import Raven.Adapt
import Raven.P4estTypes
using Raven.StaticArrays
using Raven.SparseArrays
using Test
using Aqua
using SafeTestsets

Aqua.test_all(Raven; stale_deps = (ignore = [:Requires],))

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
            cmd =
                `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --startup-file=no --project=$tmp_project $(joinpath(test_dir, f))`
            @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
        end
    end
end

MPI.Initialized() || MPI.Init()

include("arrays.jl")
include("communication.jl")
include("facecode.jl")
include("gridnumbering.jl")
include("orientation.jl")
include("sparsearrays.jl")

@testset "Balance Laws" begin
    @testset "Advection" begin
        @safetestset "entropy conservation 1d" begin
            include("balancelaws/advection/entropy_conservation_1d.jl")
        end
        @safetestset "wave 1d" begin
            include("balancelaws/advection/wave_1d.jl")
        end
        @safetestset "wave 2d" begin
            include("balancelaws/advection/wave_2d.jl")
        end
        @safetestset "wave 3d" begin
            include("balancelaws/advection/wave_3d.jl")
        end
    end

    @testset "Euler" begin
        @safetestset "isentropic vortex" begin
            include("balancelaws/euler/isentropicvortex.jl")
        end
        @safetestset "entropy conservation 1d" begin
            include("balancelaws/euler/entropy_conservation_1d.jl")
        end
        @safetestset "wave 1d" begin
            include("balancelaws/euler/wave_1d.jl")
        end
        @safetestset "wave 2d" begin
            include("balancelaws/euler/wave_2d.jl")
        end
        @safetestset "wave 3d" begin
            include("balancelaws/euler/wave_3d.jl")
        end
    end

    @testset "Multilayer Shallow Water" begin
        @safetestset "well balanced 1d" begin
            include("balancelaws/multilayer_shallow_water/well_balanced_1d.jl")
        end
        @safetestset "manufactured 1d" begin
            include("balancelaws/multilayer_shallow_water/manufactured_1d.jl")
        end
    end
end

include("testsuite.jl")

Testsuite.testsuite(Array, Float64)
Testsuite.testsuite(Array, BigFloat)

if CUDA.functional()
    @info "Running test suite with CUDA"
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Testsuite.testsuite(CuArray, Float32)
end

@testset "examples" begin
    base_dir = joinpath(@__DIR__, "..")

    test_examples = abspath.(
        joinpath.(
            base_dir,
            [
                "examples/advection",
                "examples/balancelaws/euler",
                "examples/balancelaws/euler_gravity",
                "examples/balancelaws/shallow_water",
            ],
        ),
    )

    for example_dir in test_examples
        @testset "$example_dir" begin
            mktempdir() do tmp_dir
                # Change to temporary directory so that any files created by the
                # example get cleaned up after execution.
                cd(tmp_dir)
                example_project = Pkg.Types.projectfile_path(example_dir)
                tmp_project = Pkg.Types.projectfile_path(tmp_dir)
                cp(example_project, tmp_project)

                for script in
                    filter!(s -> endswith(s, ".jl"), readdir(example_dir, join = true))
                    cmd =
                        `$(Base.julia_cmd()) --startup-file=no --project=$tmp_project -e "import Pkg; Pkg.develop(path=raw\"$base_dir\"); Pkg.instantiate()"`
                    @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
                    cmd =
                        `$(mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=$tmp_project -e "const _testing = true; include(raw\"$script\")"`
                    @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
                end
            end
        end
    end
end

runmpitests()
