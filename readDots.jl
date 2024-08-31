cd(@__DIR__);
println(pwd());

using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

using FileIO;
using Plots;

# {"\[Tau]dwTot", "CrsTot", "NTUhRef", "\[Epsilon]Sol"}

dotsFile = open("Dots.sdt")
dotsContent = read(dotsFile, String)
dotsContent = replace(dotsContent, ", \r\n" => ", ")
dotsContent = replace(dotsContent, "*^" => "e")
dotsLines = split.(chop.(split(dotsContent, "\r\n"), head=1, tail=1), ", ")

const nDots = length(dotsLines)
const nVar = length(first(dotsLines)) - 1
const nTrain = 6400
const nTest = 1600

@assert nDots >= nTrain + nTest

dotsX = Array{Float64,2}(undef, nVar, nTrain)
dotsY = Vector{Float64}(undef, nTrain)

for i ∈ 1:nTrain
    dotsX[:, i] = parse.(Float64, dotsLines[i][1:end-1])
    dotsY[i] = parse(Float64, dotsLines[i][end])
end

offset = nTrain

dotsX_test = Array{Float64,2}(undef, nVar, nTest)
dotsY_test = Vector{Float64}(undef, nTest)

for i ∈ 1:nTest
    dotsX_test[:, i] .= parse.(Float64, dotsLines[i+offset][1:end-1])
    dotsY_test[i] = parse(Float64, dotsLines[i+offset][end])
end

pScatter = plot(title="Effectiveness-NTU", dp=1000)

scatter!(dotsX[end, :], dotsY, label="Train Data", xlabel="NTU [-]", ylabel="Effectiveness [-]", markersize=2)
scatter!(dotsX_test[end, :], dotsY_test, label="Test Data", xlabel="NTU [-]", ylabel="Effectiveness [-]", markersize=2)

savefig(pScatter, "Images/Scatter_eNTU.pdf")

include("HTSR.jl");

using SymbolicRegression
using SymbolicUtils
using .HTSR

inv(x) = 1 / x
options = Options(
    binary_operators=(+, *, ^),
    unary_operators=(inv, -, log, exp),
    batching=true,
    ncyclesperiteration=512,
    maxsize=30,
    populations=50,
    population_size=75
)


const dataPath = "Data/"
const var3 = (name="Case3", variables=Dict(reverse.(enumerate([:ϵ :NTU :Cr :τ]))), sampler=nothing, op=nothing)

data = Data(dotsX, dotsY)
data_Test = Data(dotsX_test, dotsY_test)

trees, complexity = calculateSR(data, 1024, options)

saveTrees(input, (trees, complexity), n, dataPath, "Exact")
