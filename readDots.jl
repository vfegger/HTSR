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

function plotScatter(name, index, X, Y, X_T, Y_T)
    pScatter = plot(title="Effectiveness-" * name, dp=1000)

    scatter!(X[index, :], Y, label="Train Data", xlabel=name * " [-]", ylabel="Effectiveness [-]", markersize=2)
    scatter!(X_T[index, :], Y_T, label="Test Data", xlabel=name * "NTU [-]", ylabel="Effectiveness [-]", markersize=2)

    savefig(pScatter, "Images/Scatter_e" * name * ".pdf")
end

function plotScatter3D(name1, name2, index1, index2, X, Y, X_T, Y_T)
    pScatter = plot(title="Effectiveness-" * name1 * "-" * name2, dp=1000)

    scatter!(X[index1, :], X[index2, :], Y, label="Train Data", xlabel=name1 * " [-]", ylabel=name2 * " [-]", zlabel="Effectiveness [-]", markersize=2)
    scatter!(X_T[index1, :], X_T[index2, :], Y_T, label="Test Data", xlabel=name1 * "NTU [-]", ylabel=name2 * " [-]", zlabel="Effectiveness [-]", markersize=2)

    savefig(pScatter, "Images/Scatter_e" * name1 * name2 * ".pdf")
end


plotScatter("Tau", 1, dotsX, dotsY, dotsX_test, dotsY_test)
plotScatter("Cr", 2, dotsX, dotsY, dotsX_test, dotsY_test)
plotScatter("NTU", 3, dotsX, dotsY, dotsX_test, dotsY_test)

plotScatter3D("Tau", "Cr", 1, 2, dotsX, dotsY, dotsX_test, dotsY_test)
plotScatter3D("Cr", "NTU", 2, 3, dotsX, dotsY, dotsX_test, dotsY_test)
plotScatter3D("Tau", "NTU", 1, 3, dotsX, dotsY, dotsX_test, dotsY_test)


include("HTSR.jl");

using SymbolicRegression
using SymbolicUtils
using .HTSR

using JLD2


function format(path, name, n, suffix)
    return path * name * "_" * string(n) * suffix * ".jdl"
end

function saveTrees(var, trees, n::Integer, path, suffix="")
    save_object(format(path, var.name, n, "Trees" * suffix), trees)
end
function loadTrees(var, n::Integer, path, suffix="")
    return load_object(format(path, var.name, n, "Trees" * suffix))
end
function existTrees(var, n::Integer, path, suffix="")
    return isfile(format(path, var.name, n, "Trees" * suffix))
end

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

n = 1024

trees, complexity = calculateSR(data, n, options)

saveTrees(var3, (trees, complexity), n, dataPath)
