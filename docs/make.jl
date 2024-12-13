using Raven
using Documenter

DocMeta.setdocmeta!(Raven, :DocTestSetup, :(using Raven); recursive = true)

# Generate examples
include("generate.jl")

makedocs(;
    modules = [Raven],
    authors = "Raven contributors",
    repo = "https://github.com/HorribleSanity/Raven.jl/blob/{commit}{path}#{line}",
    sitename = "Raven.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://HorribleSanity.github.io/Raven.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md"
        "Usage" => "usage.md"
        "Examples" => GENERATEDEXAMPLES
        "Reference" => "reference.md"
        "Index" => "refindex.md"
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/HorribleSanity/Raven.jl",
    devbranch = "master",
    push_preview = true,
)
