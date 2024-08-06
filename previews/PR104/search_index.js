var documenterSearchIndex = {"docs":
[{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"reference/#API-reference","page":"Reference","title":"API reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"CurrentModule = Raven","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [Raven]","category":"page"},{"location":"reference/#Raven.GridArray","page":"Reference","title":"Raven.GridArray","text":"GridArray{T,N,A,G,F,L,C,D,W} <: AbstractArray{T,N}\n\nN-dimensional array of values of type T for each grid point using a struct-of-arrays like format that is GPU friendly.  Type T is assumed to be a hierarchical struct that can be flattened into an NTuple{L,E<:Real}.\n\nThe backing data array will be of type A{E} and will have the fields of T indexed via index F.\n\nGridArray also stores values for the ghost cells of the grid which are accessible if G==true.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Raven.GridArray-Union{Tuple{T}, Tuple{UndefInitializer, Raven.Grid}} where T","page":"Reference","title":"Raven.GridArray","text":"GridArray{T}(undef, grid::Grid)\n\nCreate an array containing elements of type T for each point in the grid (including the ghost cells).  The dimensions of the array is (size(referencecell(grid))..., length(grid)) as the ghost cells are hidden by default.\n\nThe type T is assumed to be able to be interpreted into an NTuple{M,L}. Some example types (some using StructArrays) are:\n\nT = NamedTuple{(:E,:B),Tuple{SVector{3,ComplexF64},SVector{3,ComplexF64}}}\nT = NTuple{5,Int64}\nT = SVector{5,Int64}\nT = ComplexF32\nT = Float32\n\nInstead of using an array-of-struct style storage, a GPU efficient struct-of-arrays like storage is used.  For example, instead of storing data like\n\njulia> T = Tuple{Int,Int};\njulia> data = Array{T}(undef, 3, 4, 2); a .= Ref((1,2))\n3×4 Matrix{Tuple{Int64, Int64}}:\n (1, 2)  (1, 2)  (1, 2)  (1, 2)\n (1, 2)  (1, 2)  (1, 2)  (1, 2)\n (1, 2)  (1, 2)  (1, 2)  (1, 2)\n\nthe data would be stored in the order\n\njulia> permutedims(reinterpret(reshape, Int, data), (2,3,1,4))\n3×4×2×2 Array{Int64, 4}:\n[:, :, 1, 1] =\n 1  1  1  1\n 1  1  1  1\n 1  1  1  1\n\n[:, :, 2, 1] =\n 2  2  2  2\n 2  2  2  2\n 2  2  2  2\n\n[:, :, 1, 2] =\n 1  1  1  1\n 1  1  1  1\n 1  1  1  1\n\n[:, :, 2, 2] =\n 2  2  2  2\n 2  2  2  2\n 2  2  2  2\n\nFor a GridArray the indices before the ones associated with T (the first two in the example above) are associated with the degrees-of-freedoms of the cells.  The one after is associated with the number of cells.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.abaqusmeshimport-Tuple{String}","page":"Reference","title":"Raven.abaqusmeshimport","text":"function AbaqusMeshImport(filename::String)\nThis function will parse an abaqus (.inp) file of 2D or 3D mesh data.\nSuch meshes are generated with HOHQMesh.jl.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.arraytype-Union{Tuple{GridArray{T, N, A}}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A}","page":"Reference","title":"Raven.arraytype","text":"arraytype(A::GridArray) -> DataType\n\nReturns the DataType used to store the data, e.g., Array or CuArray.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.comm-Tuple{GridArray}","page":"Reference","title":"Raven.comm","text":"comm(A::GridArray) -> MPI.Comm\n\nMPI communicator used by A.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.components-Union{Tuple{GridArray{T, N, A, G, F}}, Tuple{F}, Tuple{G}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A, G, F}","page":"Reference","title":"Raven.components","text":"components(A::GridArray{T})\n\nSplits A into a tuple of GridArrays where there is one for each component of T.\n\nNote, the data for the components is shared with the original array.\n\nFor example, if A isa GridArray{SVector{3, Float64}} then a tuple of type NTuple{3, GridArray{Float64}} would be returned.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.constructorof","page":"Reference","title":"Raven.constructorof","text":"constructorof(T::Type) -> constructor\n\nReturn an object constructor that can be used to construct objects of type T from their field values. Typically, constructor will be the type T with all parameters removed:\n\njulia> struct T{A,B}\n           a::A\n           b::B\n       end\n\njulia> Raven.constructorof(T{Int,Int})\nT\n\n\nThe returned constructor is used to unflatten objects hierarchical types from a list of their values. For example, in this case T(1,2) constructs an object where T.a==1 an T.b==2.\n\nThe method constructorof should be defined for types that are not constructed from a tuple of their values.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Raven.cubeshell2dgrid-Tuple{Real}","page":"Reference","title":"Raven.cubeshell2dgrid","text":"function cubeshell2dgrid(R::Real)\n\nThis function will construct the CoarseGrid of a cube shell of radius R. \nA cube shell is a 2D connectivity.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.fieldindex-Union{Tuple{GridArray{T, N, A, G, F}}, Tuple{F}, Tuple{G}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A, G, F}","page":"Reference","title":"Raven.fieldindex","text":"fieldindex(A::GridArray{T})\n\nReturns the index used in A.data to store the fields of T.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.fieldslength-Union{Tuple{GridArray{T, N, A, G, F, L}}, Tuple{L}, Tuple{F}, Tuple{G}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A, G, F, L}","page":"Reference","title":"Raven.fieldslength","text":"fieldslength(A::GridArray{T})\n\nReturns the number of fields used to store T.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.flatten-Union{Tuple{U}, Tuple{T}, Tuple{T, Type{U}}} where {T, U}","page":"Reference","title":"Raven.flatten","text":"flatten(obj, use=Real)\n\nFlattens a hierarchical type to a tuple with elements of type use.\n\nExamples\n\njulia> flatten((a=(Complex(1, 2), 3), b=4))\n(1, 2, 3, 4)\n\n\nTo convert the tuple to a vector, simply use [flatten(x)...], or using static arrays to avoid allocations: SVector(flatten(x)).\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.get_backend-Tuple{Any}","page":"Reference","title":"Raven.get_backend","text":"get_backend(::Type{T}) -> Type\n\nReturns the KernelAbstractions backend to use with kernels where A is an argument.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.get_backend-Union{Tuple{GridArray{T, N, A}}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A}","page":"Reference","title":"Raven.get_backend","text":"get_backend(A::GridArray) -> KernelAbstractions.Backend\n\nReturns the KernelAbstractions.Backend used to launch kernels interacting with A.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.materializequadranttoglobalid-Tuple{Any, Any}","page":"Reference","title":"Raven.materializequadranttoglobalid","text":"materializequadranttoglobalid(forest, ghost)\n\nGenerate the global ids for quadrants in the forest and the ghost layer.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.numbercontiguous-Union{Tuple{T}, Tuple{Type{T}, Any}} where T","page":"Reference","title":"Raven.numbercontiguous","text":"numbercontiguous(T, A; by = identity)\n\nRenumbers A contiguously in an Array{T} and returns it. The function by is a mapping for the elements of A used during element comparison, similar to sort.\n\nExamples\n\njulia> Raven.numbercontiguous(Int32, [13, 4, 5, 1, 5])\n5-element Vector{Int32}:\n 4\n 2\n 3\n 1\n 3\n\njulia> Raven.numbercontiguous(Int32, [13, 4, 5, 1, 5]; by = x->-x)\n5-element Vector{Int32}:\n 1\n 3\n 2\n 4\n 2\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.parenttype-Tuple{Any}","page":"Reference","title":"Raven.parenttype","text":"parent_type(::Type{T}) -> Type\n\nReturns the parent array that type T wraps.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.parentwithghosts-Tuple{GridArray}","page":"Reference","title":"Raven.parentwithghosts","text":"parentwithghosts(A::GridArray)\n\nReturn the underlying \"parent array\" which includes the ghost cells.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.pin-Tuple{Type, Array}","page":"Reference","title":"Raven.pin","text":"pin(T::Type, A::Array)\n\nPins the host array A for copying to arrays of type T\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.showingghosts-Union{Tuple{GridArray{T, N, A, G}}, Tuple{G}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A, G}","page":"Reference","title":"Raven.showingghosts","text":"showingghosts(A::GridArray) -> Bool\n\nPredicate indicating if the ghost layer is accessible to A.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.sizewithghosts-Tuple{GridArray}","page":"Reference","title":"Raven.sizewithghosts","text":"sizewithghosts(A::GridArray)\n\nReturn a tuple containing the dimensions of A including the ghost cells.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.sizewithoutghosts-Tuple{GridArray}","page":"Reference","title":"Raven.sizewithoutghosts","text":"sizewithoutghosts(A::GridArray)\n\nReturn a tuple containing the dimensions of A excluding the ghost cells.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.unflatten","page":"Reference","title":"Raven.unflatten","text":"unflatten(T::Type, data, use::Type=Real)\n\nConstruct an object from Tuple or Vector data and a Type T. The data should be at least as long as the queried fields (of type use) in T.\n\nExamples\n\njulia> unflatten(Tuple{Tuple{Int,Int},Complex{Int,Int}}, (1, 2, 3, 4))\n((1, 2), 3 + 4im)\n\n\n\n\n\n\n","category":"function"},{"location":"reference/#Raven.viewwithghosts-Union{Tuple{GridArray{T, N, A, false, F, L, C, D, W}}, Tuple{W}, Tuple{D}, Tuple{C}, Tuple{L}, Tuple{F}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A, F, L, C, D, W}","page":"Reference","title":"Raven.viewwithghosts","text":"viewwithghosts(A::GridArray)\n\nReturn a GridArray with the same data as A but with the ghost cells accessible.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Raven.viewwithoutghosts-Union{Tuple{GridArray{T, N, A, true, F, L, C, D, W}}, Tuple{W}, Tuple{D}, Tuple{C}, Tuple{L}, Tuple{F}, Tuple{A}, Tuple{N}, Tuple{T}} where {T, N, A, F, L, C, D, W}","page":"Reference","title":"Raven.viewwithoutghosts","text":"viewwithoutghosts(A::GridArray)\n\nReturn a GridArray with the same data as A but with the ghost cells inaccessible.\n\n\n\n\n\n","category":"method"},{"location":"#Raven-𓅂","page":"Home","title":"Raven 𓅂","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Raven is a toolbox for adapted discontinuous spectral element discretizations of partial differential equations that supports execution on distributed manycore devices (via KernelAbstractions and MPI).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some of our previous efforts in this area resulted in Canary, Bennu, and Atum.","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Contributions are encouraged. If there are additional features you would like to use, please open an issue or pull request.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Additional examples and documentation improvements are also welcome.","category":"page"},{"location":"refindex/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"refindex/","page":"Index","title":"Index","text":"","category":"page"}]
}
