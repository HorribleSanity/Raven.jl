# generate examples
import Literate
using Pkg

GENERATEDEXAMPLES = String[]

for dir = ["advection"]
  if dir == "advection"
    EXAMPLES = tuple("semdg_advection_2d")
    EXTRA = tuple()
  end
  EXAMPLEDIR = joinpath(@__DIR__, "..", "examples", dir)
  Pkg.activate(EXAMPLEDIR)
  Pkg.instantiate()
  EXAMPLEFILES = [joinpath(EXAMPLEDIR, "$f.jl") for f in EXAMPLES]
  GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated", dir)

  mkpath(GENERATEDDIR)

  for f in EXTRA
    cp(joinpath(EXAMPLEDIR, f), joinpath(GENERATEDDIR, f); force = true)
  end

  for input in EXAMPLEFILES
    script = Literate.script(input, GENERATEDDIR)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
  end

  # remove any .vtu files in the generated dir (should not be deployed)
  cd(GENERATEDDIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
  end

  for f in EXAMPLES
    md = joinpath("examples", "generated", dir, "$f.md")

    append!(GENERATEDEXAMPLES, (md,))
  end
end
