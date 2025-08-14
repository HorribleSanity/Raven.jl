export DGSEM
export WeakForm
export FluxDifferencingForm

abstract type AbstractVolumeForm end

struct WeakForm <: AbstractVolumeForm end
struct FluxDifferencingForm{VNF,K} <: AbstractVolumeForm
    volume_numericalflux::VNF
    kernel_type::K
    function FluxDifferencingForm(
        volume_numerical_flux::VNF,
        kernel_type::Symbol = :per_dir,
    ) where {VNF}
        new{VNF,Symbol}(volume_numerical_flux, kernel_type)
    end
end

struct DGSEM{L,G,A1,A2,A3,A4,VF,SNF,C,CM,DIR}
    law::L
    grid::G
    MJ::A1
    MJI::A2
    faceMJ::A3
    auxstate::A4
    volume_form::VF
    surface_numericalflux::SNF
    comm::C
    comm_manager::CM
end

function directions(
    ::DGSEM{L,G,A1,A2,A3,A4,VF,SNF,C,CM,DIR},
) where {L,G,A1,A2,A3,A4,VF,SNF,C,CM,DIR}
    return DIR
end

Raven.referencecell(dg::DGSEM) = referencecell(dg.grid)

function Adapt.adapt_structure(to, dg::DGSEM)
    names = fieldnames(DGSEM)
    args = ntuple(j -> adapt(to, getfield(dg, names[j])), length(names))
    DGSEM{typeof.(args)...,directions(dg)}(args...)
end

function DGSEM(;
    law,
    grid,
    surface_numericalflux,
    volume_form = WeakForm(),
    directions = 1:ndims(law),
    auxstate = nothing,
    comm = MPI.COMM_WORLD,
)
    _, _, MJ = components(first(volumemetrics(grid)))
    MJI = 1 ./ MJ
    _, _, faceMJ = components(first(surfacemetrics(grid)))

    if auxstate === nothing
        auxstate = auxiliary.(Ref(law), points(grid))
        aux_comm_manager = commmanager(eltype(auxstate), nodecommpattern(grid); comm)
        start!(auxstate, aux_comm_manager)
        finish!(auxstate, aux_comm_manager)
    end

    comm_manager = commmanager(typeofstate(law), nodecommpattern(grid); comm)

    args = (
        law,
        grid,
        MJ,
        MJI,
        faceMJ,
        auxstate,
        volume_form,
        surface_numericalflux,
        comm,
        comm_manager,
    )
    DGSEM{typeof.(args)...,directions}(args...)
end
Raven.get_backend(dg::DGSEM) = Raven.get_backend(arraytype(referencecell(dg)))

function (dg::DGSEM)(dq, q, time; increment = true)

    cell = referencecell(dg)
    grid = dg.grid
    backend = Raven.get_backend(dg)
    dim = ndims(cell)
    Nq = size(cell)[1]

    @assert all(size(cell) .== Nq)
    @assert(length(eltype(q)) == numberofstates(dg.law))

    start!(q, dg.comm_manager)
    if length(dg.grid) > 0
        launch_volumeterm(dg.volume_form, dq, q, dg, time; increment)
    end

    finish!(q, dg.comm_manager)

    if length(dg.grid) > 0
        Nfp = Nq^(dim - 1)
        workgroup_face = (Nfp,)
        ndrange = (Nfp * length(grid),)
        fm = facemaps(grid)
        faceix⁻, faceix⁺ = fm.vmapM, fm.vmapP
        facenormal, _, _ = components(first(surfacemetrics(grid)))
        surfaceterm!(backend, workgroup_face)(
            dg.law,
            dq,
            viewwithghosts(q),
            Val(Raven.faceoffsets(cell)),
            dg.surface_numericalflux,
            dg.MJI,
            faceix⁻,
            faceix⁺,
            dg.faceMJ,
            facenormal,
            boundarycodes(grid),
            viewwithghosts(dg.auxstate),
            Val(directions(dg));
            ndrange,
        )
    end
end

function launch_volumeterm(::WeakForm, dq, q, dg, time; increment)
    backend = Raven.get_backend(dg)
    cell = referencecell(dg)
    Nq = size(cell)[1]
    dim = ndims(cell)
    workgroup = ntuple(i -> i <= dim ? Nq : 1, 3)
    ndrange = (length(dg.grid) * workgroup[1], Base.tail(workgroup)...)

    q = reshape(q, (:, last(size(q))))
    dq = reshape(dq, (:, last(size(dq))))
    metrics = first(volumemetrics(dg.grid))
    metrics = reshape(metrics, (:, last(size(metrics))))
    MJ = reshape(dg.MJ, (:, last(size(dg.MJ))))
    MJI = reshape(dg.MJI, (:, last(size(dg.MJI))))
    auxstate = reshape(dg.auxstate, (:, last(size(dg.auxstate))))

    volumeterm!(backend, workgroup)(
        dg.law,
        dq,
        q,
        time,
        derivatives_1d(cell)[1],
        metrics,
        MJ,
        MJI,
        auxstate,
        Val(dim),
        Val(Nq),
        Val(numberofstates(dg.law)),
        Val(increment),
        Val(directions(dg));
        ndrange,
    )
end

function launch_volumeterm(form::FluxDifferencingForm, dq, q, dg, time; increment)
    cell = referencecell(dg)
    backend = Raven.get_backend(dg)
    Nq = size(cell)[1]
    dim = ndims(cell)
    Naux = length(eltype(dg.auxstate))

    q = reshape(q, (:, last(size(q))))
    dq = reshape(dq, (:, last(size(dq))))
    metrics = first(volumemetrics(dg.grid))
    metrics = reshape(metrics, (:, last(size(metrics))))
    MJ = reshape(dg.MJ, (:, last(size(dg.MJ))))
    MJI = reshape(dg.MJI, (:, last(size(dg.MJI))))
    auxstate = reshape(dg.auxstate, (:, last(size(dg.auxstate))))

    kernel_type = form.kernel_type
    if kernel_type == :naive
        # FIXME: kernel not updated to support directions
        @assert directions(dg) == 1:ndims(dg.law)
        workgroup = ntuple(i -> i <= min(2, dim) ? Nq : 1, 3)
        ndrange = (length(dg.grid) * workgroup[1], Base.tail(workgroup)...)
        esvolumeterm!(backend, workgroup)(
            dg.law,
            dq,
            q,
            time,
            derivatives_1d(cell)[1],
            form.volume_numericalflux,
            metrics,
            MJ,
            MJI,
            auxstate,
            Val(dim),
            Val(Nq),
            Val(numberofstates(dg.law)),
            Val(Naux),
            Val(increment),
            Val(directions(dg));
            ndrange,
        )
    elseif kernel_type == :per_dir
        workgroup = ntuple(i -> i <= dim ? Nq : 1, 3)
        ndrange = (length(dg.grid) * workgroup[1], Base.tail(workgroup)...)

        for dir in directions(dg)
            esvolumeterm_dir!(backend, workgroup)(
                dg.law,
                dq,
                q,
                time,
                derivatives_1d(cell)[1],
                form.volume_numericalflux,
                metrics,
                MJ,
                MJI,
                auxstate,
                Val(dir),
                Val(dim),
                Val(Nq),
                Val(numberofstates(dg.law)),
                Val(Naux),
                Val(dir == directions(dg)[1] ? increment : true);
                ndrange,
            )
        end
    elseif kernel_type == :per_dir_symmetric
        workgroup = ntuple(i -> i <= dim ? Nq : 1, 3)
        ndrange = (length(dg.grid) * workgroup[1], Base.tail(workgroup)...)
        for dir in directions(dg)
            esvolumeterm_dir_symmetric!(backend, workgroup)(
                dg.law,
                dq,
                q,
                time,
                derivatives_1d(cell)[1],
                form.volume_numericalflux,
                metrics,
                MJ,
                MJI,
                auxstate,
                Val(dir),
                Val(dim),
                Val(Nq),
                Val(numberofstates(dg.law)),
                Val(Naux),
                Val(dir == directions(dg)[1] ? increment : true);
                ndrange,
            )
        end
    else
        error("Unknown kernel type $kernel_type")
    end
end

@kernel function volumeterm!(
    law,
    dq,
    q,
    time,
    D,
    metrics,
    MJ,
    MJI,
    auxstate,
    ::Val{dim},
    ::Val{Nq},
    ::Val{Ns},
    ::Val{increment},
    ::Val{directions},
) where {dim,Nq,Ns,increment,directions}
    @uniform begin
        FT = eltype(law)
        Nq1 = Nq
        Nq2 = dim > 1 ? Nq : 1
        Nq3 = dim > 2 ? Nq : 1
        Ndir = length(directions)
    end

    l_F = @localmem FT (Nq1, Nq2, Nq3, Ndir, Ns)
    dqijk = @private FT (Ns,)
    p_MJ = @private FT (1,)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq * (j - 1 + Nq * (k - 1))

        g = metrics[ijk, e].dRdX

        qijk = q[ijk, e]
        auxijk = auxstate[ijk, e]
        MJijk = MJ[ijk, e]

        fijk = flux(law, qijk, auxijk)

        @unroll for s = 1:Ns
            @unroll for d = 1:Ndir
                dir = directions[d]
                l_F[i, j, k, d, s] = 0
                @unroll for dd = 1:dim
                    l_F[i, j, k, d, s] += g[dir, dd] * fijk[dd, s]
                end
                l_F[i, j, k, d, s] *= MJijk
            end
        end

        @unroll for s = 1:Ns
            dqijk[s] = -zero(FT)
        end
        source!(law, dqijk, qijk, auxijk, dim, directions, time)
        source!(law, problem(law), dqijk, qijk, auxijk, dim, directions, time)
        nonconservative_term!(law, dqijk, qijk, auxijk, directions, dim)

        @synchronize

        ijk = i + Nq * (j - 1 + Nq * (k - 1))
        MJIijk = MJI[ijk, e]

        @unroll for n = 1:Nq
            Dni = D[n, i] * MJIijk
            Dnj = D[n, j] * MJIijk
            Dnk = D[n, k] * MJIijk
            @unroll for s = 1:Ns
                dir = 1
                if 1 ∈ directions
                    dqijk[s] += Dni * l_F[n, j, k, dir, s]
                    dir += 1
                end
                if 2 ∈ directions
                    dqijk[s] += Dnj * l_F[i, n, k, dir, s]
                    dir += 1
                end
                if 3 ∈ directions
                    dqijk[s] += Dnk * l_F[i, j, n, dir, s]
                end
            end
        end

        if increment
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end
    end
end

@kernel function esvolumeterm!(
    law,
    dq,
    q,
    time,
    D,
    volume_numericalflux,
    metrics,
    MJ,
    MJI,
    auxstate,
    ::Val{dim},
    ::Val{Nq},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment},
    ::Val{directions},
) where {dim,Nq,Ns,Naux,increment,directions}
    @uniform begin
        FT = eltype(law)
        Nq1 = Nq
        Nq2 = dim > 1 ? Nq : 1
        Nq3 = dim > 2 ? Nq : 1

        q1 = MVector{Ns,FT}(undef)
        q2 = MVector{Ns,FT}(undef)
        aux1 = MVector{Naux,FT}(undef)
        aux2 = MVector{Naux,FT}(undef)
    end

    dqijk = @private FT (Ns,)

    pencil_q = @private FT (Ns, Nq3)
    pencil_aux = @private FT (Naux, Nq3)
    pencil_g3 = @private FT (3, Nq3)
    pencil_MJ = @private FT (Nq3,)

    l_q = @localmem FT (Nq1, Nq2, Ns)
    l_aux = @localmem FT (Nq1, Nq2, Naux)
    l_g = @localmem FT (Nq1, Nq2, min(2, dim), dim)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for k = 1:Nq3
            ijk = i + Nq * (j - 1 + Nq * (k - 1))

            pencil_MJ[k] = MJ[ijk, e]
            @unroll for s = 1:Ns
                pencil_q[s, k] = q[ijk, e][s]
            end
            @unroll for d in directions
                pencil_aux[d, k] = auxstate[ijk, e][d]
            end
            if dim > 2
                @unroll for d = 1:dim
                    pencil_g3[d, k] = metrics[ijk, e].dRdX[3, d]
                    pencil_g3[d, k] *= pencil_MJ[k]
                end
            end
        end

        @unroll for k = 1:Nq3
            @synchronize
            ijk = i + Nq * (j - 1 + Nq * (k - 1))

            @unroll for s = 1:Ns
                l_q[i, j, s] = pencil_q[s, k]
            end
            @unroll for s = 1:Naux
                l_aux[i, j, s] = pencil_aux[s, k]
            end

            MJk = pencil_MJ[k]
            @unroll for d = 1:dim
                l_g[i, j, 1, d] = MJk * metrics[ijk, e].dRdX[1, d]
                if dim > 1
                    l_g[i, j, 2, d] = MJk * metrics[ijk, e].dRdX[2, d]
                end
            end

            @synchronize

            @unroll for s = 1:Ns
                dqijk[s] = -zero(FT)
            end

            @unroll for s = 1:Ns
                q1[s] = l_q[i, j, s]
            end
            @unroll for s = 1:Naux
                aux1[s] = l_aux[i, j, s]
            end

            source!(law, dqijk, q1, aux1, dim, directions, time)
            source!(law, problem(law), dqijk, q1, aux1, dim, directions, time)

            MJIijk = 1 / pencil_MJ[k]
            @unroll for n = 1:Nq
                @unroll for s = 1:Ns
                    q2[s] = l_q[n, j, s]
                end
                @unroll for s = 1:Naux
                    aux2[s] = l_aux[n, j, s]
                end

                f = twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)
                @unroll for s = 1:Ns
                    Din = MJIijk * D[i, n]
                    Dni = MJIijk * D[n, i]
                    @unroll for d = 1:dim
                        dqijk[s] -= Din * l_g[i, j, 1, d] * f[d, s]
                        dqijk[s] += f[d, s] * l_g[n, j, 1, d] * Dni
                    end
                end

                if dim > 1
                    @unroll for s = 1:Ns
                        q2[s] = l_q[i, n, s]
                    end
                    @unroll for s = 1:Naux
                        aux2[s] = l_aux[i, n, s]
                    end
                    f = twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)
                    @unroll for s = 1:Ns
                        Djn = MJIijk * D[j, n]
                        Dnj = MJIijk * D[n, j]
                        @unroll for d = 1:dim
                            dqijk[s] -= Djn * l_g[i, j, 2, d] * f[d, s]
                            dqijk[s] += f[d, s] * l_g[i, n, 2, d] * Dnj
                        end
                    end
                end

                if dim > 2
                    @unroll for s = 1:Ns
                        q2[s] = pencil_q[s, n]
                    end
                    @unroll for s = 1:Naux
                        aux2[s] = pencil_aux[s, n]
                    end
                    f = twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)
                    @unroll for s = 1:Ns
                        Dkn = MJIijk * D[k, n]
                        Dnk = MJIijk * D[n, k]
                        @unroll for d = 1:dim
                            dqijk[s] -= Dkn * pencil_g3[d, k] * f[d, s]
                            dqijk[s] += f[d, s] * pencil_g3[d, n] * Dnk
                        end
                    end
                end
            end

            ijk = i + Nq * (j - 1 + Nq * (k - 1))
            if increment
                dq[ijk, e] += dqijk
            else
                dq[ijk, e] = dqijk[:]
            end
        end
    end
end

@kernel function esvolumeterm_dir!(
    law,
    dq,
    q,
    time,
    D,
    volume_numericalflux,
    metrics,
    MJ,
    MJI,
    auxstate,
    ::Val{dir},
    ::Val{dim},
    ::Val{Nq},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment},
) where {dir,dim,Nq,Ns,Naux,increment}
    @uniform begin
        FT = eltype(law)
        Nq1 = Nq
        Nq2 = dim > 1 ? Nq : 1
        Nq3 = dim > 2 ? Nq : 1
    end

    dqijk = @private FT (Ns,)

    q1 = @private FT (Ns,)
    aux1 = @private FT (Naux,)

    l_g = @localmem FT (Nq^3, 3)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq * (j - 1 + Nq * (k - 1))

        MJijk = MJ[ijk, e]
        @unroll for d = 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].dRdX[dir, d]
        end

        @unroll for s = 1:Ns
            dqijk[s] = -zero(FT)
        end

        @unroll for s = 1:Ns
            q1[s] = q[ijk, e][s]
        end
        @unroll for s = 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        source!(law, dqijk, q1, aux1, dim, (dir,), time)
        source!(law, problem(law), dqijk, q1, aux1, dim, (dir,), time)

        @synchronize

        ijk = i + Nq * (j - 1 + Nq * (k - 1))

        MJIijk = MJI[ijk, e]
        @unroll for n = 1:Nq
            if dir == 1
                id = i
                ild = n + Nq * ((j - 1) + Nq * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq * ((n - 1) + Nq * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq * ((j - 1) + Nq * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]

            f = twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)
            @unroll for s = 1:Ns
                Ddn = MJIijk * D[id, n]
                Dnd = MJIijk * D[n, id]
                @unroll for d = 1:dim
                    dqijk[s] -= Ddn * l_g[ijk, d] * f[d, s]
                    dqijk[s] += f[d, s] * l_g[ild, d] * Dnd
                end
            end
        end

        if increment
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end
    end
end

@kernel function esvolumeterm_dir_symmetric!(
    law,
    dq,
    q,
    time,
    D,
    volume_numericalflux,
    metrics,
    MJ,
    MJI,
    auxstate,
    ::Val{dir},
    ::Val{dim},
    ::Val{Nq},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment},
) where {dir,dim,Nq,Ns,Naux,increment}
    @uniform begin
        FT = eltype(law)
        Nq1 = Nq
        Nq2 = dim > 1 ? Nq : 1
        Nq3 = dim > 2 ? Nq : 1
    end

    dqijk1 = @private FT (Ns,)
    dqijk2 = @private FT (Ns,)

    q1 = @private FT (Ns,)
    aux1 = @private FT (Naux,)

    l_g = @localmem FT (Nq^3, 3)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq * (j - 1 + Nq * (k - 1))

        MJijk = MJ[ijk, e]
        @unroll for d = 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].dRdX[dir, d]
        end

        @unroll for s = 1:Ns
            dqijk1[s] = -zero(FT)
            dqijk2[s] = -zero(FT)
        end

        @unroll for s = 1:Ns
            q1[s] = q[ijk, e][s]
        end
        @unroll for s = 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        source!(law, dqijk1, q1, aux1, dim, (dir,), time)
        source!(law, problem(law), dqijk1, q1, aux1, dim, (dir,), time)

        @synchronize

        ijk = i + Nq * (j - 1 + Nq * (k - 1))

        MJIijk = MJI[ijk, e]
        @unroll for n = 1:Nq
            if dir == 1
                id = i
                ild = n + Nq * ((j - 1) + Nq * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq * ((n - 1) + Nq * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq * ((j - 1) + Nq * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]

            f = twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)
            @unroll for s = 1:Ns
                Ddn = MJIijk * D[id, n]
                Dnd = MJIijk * D[n, id]
                @unroll for d = 1:dim
                    dqijk1[s] -= Ddn * l_g[ijk, d] * f[d, s] / 2
                    dqijk1[s] += f[d, s] * l_g[ild, d] * Dnd / 2
                end
            end
        end
        @unroll for n = Nq:-1:1
            if dir == 1
                id = i
                ild = n + Nq * ((j - 1) + Nq * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq * ((n - 1) + Nq * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq * ((j - 1) + Nq * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]

            f = twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)
            @unroll for s = 1:Ns
                Ddn = MJIijk * D[id, n]
                Dnd = MJIijk * D[n, id]
                @unroll for d = 1:dim
                    dqijk2[s] -= Ddn * l_g[ijk, d] * f[d, s] / 2
                    dqijk2[s] += f[d, s] * l_g[ild, d] * Dnd / 2
                end
            end
        end

        if increment
            dq[ijk, e] += (dqijk1 + dqijk2)
        else
            dq[ijk, e] = (dqijk1+dqijk2)[:]
        end
    end
end

@kernel function surfaceterm!(
    law,
    dq,
    q,
    ::Val{faceoffsets},
    numericalflux,
    MJI,
    faceix⁻,
    faceix⁺,
    faceMJ,
    facenormal,
    boundaryfaces,
    auxstate,
    ::Val{directions},
) where {faceoffsets,directions}
    @uniform begin
        FT = eltype(q)
    end

    e⁻ = @index(Group, Linear)
    i = @index(Local, Linear)

    @inbounds begin
        @unroll for d in directions
            @unroll for f = 1:2
                face = 2(d - 1) + f
                j = i + faceoffsets[face]
                id⁻ = faceix⁻[j, e⁻]

                n⃗ = facenormal[j, e⁻]
                fMJ = faceMJ[j, e⁻]

                aux⁻ = auxstate[id⁻]
                q⁻ = q[id⁻]

                boundarytag = boundaryfaces[face, e⁻]
                if boundarytag == 0
                    id⁺ = faceix⁺[j, e⁻]
                    q⁺ = q[id⁺]
                    aux⁺ = auxstate[id⁺]
                else
                    q⁺, aux⁺ = boundarystate(law, problem(law), n⃗, q⁻, aux⁻, boundarytag)
                end

                nf = surfaceflux(numericalflux, law, n⃗, q⁻, aux⁻, q⁺, aux⁺)
                dq[id⁻] -= fMJ * nf * MJI[id⁻]

                @synchronize(mod(face, 2) == 0)
            end
        end
    end
end
