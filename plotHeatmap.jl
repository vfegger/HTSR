cd(@__DIR__);
println(pwd());

using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

include("HTSR.jl");

using IterTools
using SymbolicRegression
using SymbolicUtils
using .HTSR

using Plots
using LaTeXStrings

using Glob
using JLD2

const var1 = (name="Case1", variables=Dict(reverse.(enumerate([:Re :Pr]))), range=Dict([(:Re, (min=1.0e2, max=5.0e5)), (:Pr, (min=0.5e0, max=1.0e2))]), op=((Re::Number, Pr::Number) -> 0.663e0 * Re^(1.0e0 / 2.0e0) * Pr^(1.0f0 / 3.0f0)))
const var2 = (name="Case2", variables=Dict(reverse.(enumerate([:Ra :Pr]))), range=Dict([(:Ra, (min=1.0e2, max=1.0e8)), (:Pr, (min=0.5e0, max=1.0e2))]), op=((Ra::Number, Pr::Number) -> 0.677e0 * ((2.0e1 / (2.1e1 * Pr)) + 1.0e0)^(-1.0e0 / 4.0e0) * Ra^(1.0e0 / 4.0e0)))
const var3 = (name="Case3", variables=Dict(reverse.(enumerate([:ϵ :NTU]))), range=nothing, op=nothing)

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

function format(path, name, n, suffix)
    return path * name * "_" * string(n) * suffix * ".jdl"
end

function loadTrees(var, n::Integer, path, suffix="")
    return load_object(format(path, var.name, n, "Trees" * suffix))
end

function plotHeatmap(var, n::Integer, resolution::Integer)

    trees, complexity = loadTrees(var, 1600, "Data/")

    X = Array{Float64,2}(undef, length(var.variables), resolution^length(var.variables))
    values = [range(var.range[s].min, var.range[s].max, resolution) for s in keys(var.variables)]
    combination = collect(IterTools.product(values...))
    for i in 1:resolution^length(var.variables)
        X[:, i] .= combination[i]
    end

    Y = [reshape(eval_tree_array(tree, X, options)[1], (resolution, resolution)) for tree in trees]
    z = reshape(var.op.(X[1, :], X[2, :]), (resolution, resolution))

    pltResult = [heatmap(values[1], values[2], y) for y in Y]
    pltExact = heatmap(values[1], values[2], z)

    return pltResult, pltExact
end