Stream(_) = nothing
synchronize(backend, _) = KernelAbstractions.synchronize(backend)
stream!(f::Function, _, _) = f()
