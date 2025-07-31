abstract type AbstractMeshImport end

struct AbaqusMeshImport{N,C,F,I,B,D} <: AbstractMeshImport
    nodes::Any
    connectivity::Any
    face_iscurved::Any
    face_degree::Int
    face_interpolation::Any
    type_boundary::Any
    type_boundary_map::Any
end

dimensions(a::AbaqusMeshImport) = length(a.nodes[1])
interpolation_degree(a::AbaqusMeshImport) = a.face_degree
interpolation(a::AbaqusMeshImport) = a.face_interpolation
interpolationdegree(a::AbaqusMeshImport) = a.face_degree

function abaqusmeshimport(
    nodes,
    connectivity,
    face_iscurved,
    face_degree,
    face_interpolation,
    type_boundary,
    type_boundary_map,
)
    N, C, F, I, B, D = typeof.([
        nodes,
        connectivity,
        face_iscurved,
        face_degree,
        face_interpolation,
        type_boundary,
        type_boundary_map,
    ])
    return AbaqusMeshImport{N,C,F,I,B,D}(
        nodes,
        connectivity,
        face_iscurved,
        face_degree,
        face_interpolation,
        type_boundary,
        type_boundary_map,
    )
end

"""
    function AbaqusMeshImport(filename::String)
    This function will parse an abaqus (.inp) file of 2D or 3D mesh data.
    Such meshes are generated with HOHQMesh.jl.
"""
function abaqusmeshimport(filename::String)
    @assert isfile(filename) "File does not exist"
    file = open(filename)
    filelines = readlines(file)

    # indicate the line with the section header
    nodes_linenum = findline("*NODE", filelines)
    elements_linenum = findline("*ELEMENT", filelines)
    curvature_linenum = findline("HOHQMesh boundary information", filelines)
    node_count = elements_linenum - nodes_linenum - 1
    element_count = curvature_linenum - elements_linenum - 1
    type_boundary_linenum = length(filelines) - element_count + 1


    @assert 0 < nodes_linenum < elements_linenum < curvature_linenum "Improper abaqus file"
    @assert findline("mesh polynomial degree", filelines) == curvature_linenum + 1 "Missing Poly degree"

    # degree of interpolant along curved edges.
    face_degree = parse(Int8, split(filelines[curvature_linenum+1], " = ")[2])

    type = split(filelines[elements_linenum], r", [a-zA-Z]*=")[2]
    # CPS4 are 2D quads C3D8 are 3D hexs
    @assert type == "CPS4" || type == "C3D8"
    dims = (type == "CPS4") ? 2 : 3
    FT = Float64
    IT = Int
    # Extract node coords
    nodes = Vector{SVector{dims,FT}}(undef, node_count)
    for nodeline in filelines[(nodes_linenum+1):(elements_linenum-1)]
        temp_data = split(nodeline, ", ")
        nodenumber = parse(IT, temp_data[1])
        nodes[nodenumber] = SVector{dims,FT}(parse.(FT, temp_data[2:(dims+1)]))
    end

    # Extract element coords
    # HOHQMesh uses right hand rule for node numbering. We need Z order perm will reorder
    nodes_per_element = (dims == 2) ? 4 : 8
    perm = (dims == 2) ? [1, 2, 4, 3] : [1, 2, 4, 3, 5, 6, 8, 7]
    connectivity = Vector{NTuple{nodes_per_element,IT}}(undef, element_count)
    for elementline in filelines[(elements_linenum+1):(curvature_linenum-1)]
        temp_data = split(elementline, ", ")
        elementnumber = parse(IT, temp_data[1])
        connectivity[elementnumber] = Tuple(parse.(IT, temp_data[2:end]))[perm]
    end

    # Extract which edges are curved
    scanningidx = curvature_linenum + 2
    faces_per_element = (dims == 2) ? 4 : 6
    face_iscurved = Vector{NTuple{faces_per_element,IT}}(undef, element_count)
    for i = 1:element_count
        scanningidx += 1
        face_iscurved[i] = Tuple(parse.(IT, split(filelines[scanningidx])[2:end]))
        scanningidx += sum(face_iscurved[i]) * (face_degree + 1)^(dims - 1) + 1
    end

    # Extract interpolation nodes of curved edges
    face_interpolation = Vector{NTuple{dims,Float64}}(
        undef,
        sum(sum.(face_iscurved)) * (face_degree + 1)^(dims - 1),
    )
    scanningidx = curvature_linenum + 2
    idx = 1
    for i = 1:element_count
        scanningidx += 2
        for node = 1:(sum(face_iscurved[i])*(face_degree+1)^(dims-1))
            face_interpolation[idx] =
                Tuple(parse.(FT, split(filelines[scanningidx])[2:(dims+1)]))
            idx += 1
            scanningidx += 1
        end
    end

    # Extract type_boundary data
    type_boundary = Vector{NTuple{faces_per_element,Int64}}(undef, element_count)

    # We will use integers to label boundary conditions on each element face for GPU compatibility.
    # HOHQMesh uses strings so here we create the mapping. zero corresponds an face which is not along a boundary
    # as does "---" with HOHQMesh
    temp_key = Vector{String}(["---"])
    for line in filelines[type_boundary_linenum:end]
        append!(temp_key, split(line)[2:end])
    end

    key = unique(temp_key)
    type_boundary_map = Dict(zip(key, 0:(length(key)-1)))
    for (element, line) in enumerate(filelines[type_boundary_linenum:end])
        data = split(line)[2:end]
        type_boundary[element] = Tuple([type_boundary_map[word] for word in data])
    end

    return abaqusmeshimport(
        nodes,
        connectivity,
        face_iscurved,
        face_degree,
        face_interpolation,
        type_boundary,
        type_boundary_map,
    )
end

function findline(word::String, filelines::Vector{String})
    for (i, line) in enumerate(filelines)
        if occursin(word, line)
            return i
        end
    end
    return -1
end
