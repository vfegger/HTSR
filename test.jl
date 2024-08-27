cd(@__DIR__);
println(pwd());

using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

include("HTSR.jl");

using Random
using Distributions
using SymbolicRegression
using SymbolicUtils
using .HTSR

using Plots
using LaTeXStrings

using Glob
using JLD2
using BenchmarkTools

clean = false

const rseed::Integer = 1234
Random.seed!(rseed)

const scale::Integer = 100
const growth::Integer = 2
const nsamples::Integer = 8
const ntest::Integer = 100000
const ncpi::Integer = 516
const niter::Integer = 1024
const complexity::Integer = 20
const ndata::Integer = 6400

const dataPath = "Data/"
const imagePath = "Images/"

if !isdir(dataPath)
    mkdir(dataPath)
end
if !isdir(imagePath)
    mkdir(imagePath)
end

const var1 = (name="Case1", variables=Dict(reverse.(enumerate([:Re :Pr]))), sampler=Dict([:Re => Uniform(1.0e2, 5.0e5), :Pr => Uniform(0.5e0, 100.0e0)]), op=((Re::Number, Pr::Number) -> 0.663e0 * Re^(1.0e0 / 2.0e0) * Pr^(1.0f0 / 3.0f0)))
const var2 = (name="Case2", variables=Dict(reverse.(enumerate([:Ra :Pr]))), sampler=Dict([:Ra => Uniform(1.0e2, 1.0e8), :Pr => Uniform(0.5e0, 100.0e0)]), op=((Ra::Number, Pr::Number) -> 0.677e0 * ((2.0e1 / (2.1e1 * Pr)) + 1.0e0)^(-1.0e0 / 4.0e0) * Ra^(1.0e0 / 4.0e0)))
const var3 = (name="Case3", variables=Dict(reverse.(enumerate([:ϵ :NTU]))), sampler=nothing, op=nothing)

if clean
    fileNames = vcat(glob("*.jdl", "./Data/"), glob("*.pdf", "./Images/"))
    foreach(rm, fileNames)
    fileNames = nothing
end
inv(x) = 1 / x
options = Options(
    binary_operators=(+, *, ^),
    unary_operators=(inv, -),
    batching=true,
    ncyclesperiteration=ncpi,
    maxsize=complexity,
    populations=50,
    population_size=75
)

function case(var, n::Integer)
    X = Array{Float64,2}(undef, length(var.variables), n)
    for (symbol, distribution) ∈ var.sampler
        rand!(distribution, @view(X[var.variables[symbol], :]))
    end

    y = var.op.(eachrow(X)...)

    return Data(X, y)
end

function format(path, name, n, suffix)
    return path * name * "_" * string(n) * suffix * ".jdl"
end

function generateData(var, n::Integer, path, suffix)
    data = case(var, n)
    save_object(format(path, var.name, n, suffix), data)
end

function loadData(var, n::Integer, path, suffix)
    fileName = format(path, var.name, n, suffix)
    if !isfile(fileName)
        generateData(var, n, path, suffix)
    end
    return load_object(fileName)
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

# Sample Dependency

function plotSampleDep(sets, data, options)
    plot_font = "Computer Modern"
    default(fontfamily=plot_font)
    plt = plot(title="Loss over Complexity", dp=1000)
    for set in sets
        residuals = [sum(abs2, eval_tree_array(tree, data.X, options)[1] - data.y) for tree in set.trees]
        num_points = length(data.y)
        loss = log10.(residuals ./ num_points)
        plot!(set.complexity, loss, label=set.label, xlabel="Complexity [-]", ylabel="Log Loss [-]")
    end
    return plt
end

function sampleRun(input, options)

    plotdata = []

    for i in 0:nsamples-1
        n = scale * growth^i
        println("Step Samples = " * string(n))

        if !existTrees(input, n, dataPath)
            data = loadData(input, n, dataPath, "")

            trees, complexity = calculateSR(data, niter, options)

            saveTrees(input, (trees, complexity), n, dataPath)
        end

        trees, complexity = loadTrees(input, n, dataPath)

        push!(plotdata, (trees=trees, complexity=complexity, label="N=$n"))
    end

    data = loadData(input, ntest, dataPath, "Test")

    return plotSampleDep(plotdata, data, options)
end

pSample1_losses = sampleRun(var1, options)
savefig(pSample1_losses, "Images/sample_losses_case1.pdf")

pSample2_losses = sampleRun(var2, options)
savefig(pSample2_losses, "Images/sample_losses_case2.pdf")

# Noise Robust Test

function plotNoiseDep(sets, data, options)
    plot_font = "Computer Modern"
    default(fontfamily=plot_font)
    plt = plot(title="Loss over Complexity", dp=1000)
    for set in sets
        residuals = [sum(abs2, eval_tree_array(tree, data.X, options)[1] - data.y) for tree in set.trees]
        num_points = length(data.y)
        loss = log10.(residuals ./ num_points)
        plot!(set.complexity, loss, label=set.label, xlabel="Complexity [-]", ylabel="Log Loss [-]")
    end
    return plt
end

function noiseRun(input, n::Integer, options)

    if existTrees(input, n, dataPath, "Exact") && existTrees(input, n, dataPath, "Noise1") && existTrees(input, n, dataPath, "Noise01") && existTrees(input, n, dataPath, "Noise001")
        data = loadData(input, n, dataPath, "")
        dataTest = loadData(input, ntest, dataPath, "Test")
        data_noise_1 = deepcopy(data)
        data_noise_01 = deepcopy(data)
        data_noise_001 = deepcopy(data)

        data_noise_1.y .+= 0.01 .* data_noise_1.y .* randn(size(data_noise_1.y))
        data_noise_01.y .+= 0.001 .* data_noise_01.y .* randn(size(data_noise_01.y))
        data_noise_001.y .+= 0.0001 .* data_noise_001.y .* randn(size(data_noise_001.y))

        trees_exact, complexity_exact = calculateSR(data, niter, options)
        trees_noise_1, complexity_noise_1 = calculateSR(data_noise_1, niter, options)
        trees_noise_01, complexity_noise_01 = calculateSR(data_noise_01, niter, options)
        trees_noise_001, complexity_noise_001 = calculateSR(data_noise_001, niter, options)

        saveTrees(input, (trees_exact, complexity_exact), n, dataPath, "Exact")
        saveTrees(input, (trees_noise_1, complexity_noise_1), n, dataPath, "Noise1")
        saveTrees(input, (trees_noise_01, complexity_noise_01), n, dataPath, "Noise01")
        saveTrees(input, (trees_noise_001, complexity_noise_001), n, dataPath, "Noise001")
    end

    trees_exact, complexity_exact = loadTrees(input, n, dataPath, "Exact")
    trees_noise_1, complexity_noise_1 = loadTrees(input, n, dataPath, "Noise1")
    trees_noise_01, complexity_noise_01 = loadTrees(input, n, dataPath, "Noise01")
    trees_noise_001, complexity_noise_001 = loadTrees(input, n, dataPath, "Noise001")

    return plotNoiseDep([
            (trees=trees_exact, complexity=complexity_exact, label="Exact"),
            (trees=trees_noise_1, complexity=complexity_noise_1, label="Noise = 1%"),
            (trees=trees_noise_01, complexity=complexity_noise_01, label="Noise = 0.1%"),
            (trees=trees_noise_001, complexity=complexity_noise_001, label="Noise = 0.01%"),
        ], dataTest, options)
end

pNoise1_losses = noiseRun(var1, ndata, options)
savefig(pNoise1_losses, "Images/losses_case1.pdf")
pNoise2_losses = noiseRun(var2, ndata, options)
savefig(pNoise2_losses, "Images/losses_case2.pdf")

fileNames = glob("hall_of_fame*")
foreach(rm, fileNames)
fileNames = nothing

exit(0)