# This file was modified from P4est.jl
using Pkg
Pkg.add("MPIPreferences")
Pkg.add("Preferences")
Pkg.add("UUIDs")

@static if VERSION >= v"1.8"
    Pkg.compat("MPIPreferences", "0.1")
    Pkg.compat("Preferences", "1")
end

const RAVEN_TEST = get(ENV, "RAVEN_TEST", "RAVEN_JLL_MPI_DEFAULT")
const RAVEN_TEST_LIBP4EST = get(ENV, "RAVEN_TEST_LIBP4EST", "")
const RAVEN_TEST_LIBSC = get(ENV, "RAVEN_TEST_LIBSC", "")

@static if RAVEN_TEST == "RAVEN_CUSTOM_MPI_CUSTOM"
    import MPIPreferences
    MPIPreferences.use_system_binary()
end

@static if RAVEN_TEST == "RAVEN_CUSTOM_MPI_CUSTOM"
    import UUIDs, Preferences
    Preferences.set_preferences!(
        UUIDs.UUID("7d669430-f675-4ae7-b43e-fab78ec5a902"), # UUID of P4est.jl
        "libp4est" => RAVEN_TEST_LIBP4EST,
        force = true,
    )
    Preferences.set_preferences!(
        UUIDs.UUID("7d669430-f675-4ae7-b43e-fab78ec5a902"), # UUID of P4est.jl
        "libsc" => RAVEN_TEST_LIBSC,
        force = true,
    )
end

@info "Raven.jl tests configured" RAVEN_TEST RAVEN_TEST_LIBP4EST RAVEN_TEST_LIBSC
