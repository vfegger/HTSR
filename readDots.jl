cd(@__DIR__);
println(pwd());

using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

using FileIO;
using Plots;

dotsFile = open("Dots.sdt") do f
    readlines(f)[2:end] |> (s -> (split.(chop.(s, head=1, tail=1), ", ")))
end

dotsValues = [[parse(Float64, vs[end-1]), parse(Float64, vs[end])] for vs in dotsFile]

plotValuesX = [val[1] for val in dotsValues]
plotValuesY = [val[2] for val in dotsValues]

scatter(plotValuesX, plotValuesY)

include("HTSR.jl");

using SymbolicRegression
using SymbolicUtils
using .HTSR

inv(x) = 1 / x
options = Options(
    binary_operators=(+, *, ^),
    unary_operators=(inv, -, log, exp),
    batching=true,
    ncyclesperiteration=500,
    maxsize=20,
    populations=50,
    population_size=75
)


const dataPath = "Data/"
const var3 = (name="Case3", variables=Dict(reverse.(enumerate([:ϵ :NTU]))), sampler=nothing, op=nothing)

data = Data(reshape(plotValuesX,(1,length(plotValuesX))), plotValuesY)

trees, complexity = calculateSR(data, 100, options)

