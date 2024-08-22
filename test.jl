cd(@__DIR__);
println(pwd());

using Pkg;
Pkg.activate(pwd());

include("HTSR.jl");

using Random
using Distributions
using SymbolicRegression
using SymbolicUtils
using .HTSR

using Plots
using LaTeXStrings

using BenchmarkTools

Random.seed!(1234)

const scale::Integer = 100
const growth::Integer = 2
const nsamples::Integer = 8
const ntest::Integer = 100000
const ncpi::Integer = 32
const complexity::Integer = 20

const var1 = (variables=[:Re :Pr], sampler=Dict([p1[1] => Uniform(1.0e2, 5.0e5), p1[2] => Uniform(0.5, 100.0)]), op=((Re::Number, Pr::Number) -> 0.663 * Re^(1.0 / 2.0) * Pr^(1.0 / 3.0)))
const var2 = (variables=[:Ra :Pr], sampler=Dict([p2[1] => Uniform(1.0e2, 1.0e8), p2[2] => Uniform(0.5, 100.0)]), op=((Ra::Number, Pr::Number) -> 0.677 * ((20.0 / (21.0 * Pr)) + 1.0)^(-0.25) * Ra^(0.25)))
const var3 = (variables=[:ϵ :NTU], sampler=nothing, op=nothing)

function sample!(X, sampler)
    @assert size(X, 1) == length(sampler)
    for (i, s) ∈ sampler
        rand!(s, @view(x[i, :]))
    end
end

function case(var, n::Integer)
    X = Array{Float64,2}(undef, length(var.variables), n)
    sample!(X, var.sampler)

    y = var.op.(eachrow(X)...)

    return Data(X, y)
end

# Sample Dependency

function plotSampleDep(sets, data, options)
    plot_font = "Computer Modern"
    default(fontfamily=plot_font)
    plt = plot(title="Loss over Complexity", dp=1000)
    for set in sets
        residuals = [sum(abs2, eval_tree_array(tree, data.X, options)[1] - data.y) for tree in set.trees]
        num_points = length(dataX)
        loss = log10.(residuals ./ num_points)
        plot!(set.complexity, loss, label=set.label, xlabel="Complexity [-]", ylabel="Log Loss [-]")
    end
    return plt
end

function SampleRun(input)

    plotdata = []

    inv(x) = 1 / x
    options = Options(
        binary_operators=(+, *, ^),
        unary_operators=(inv, -),
        batching=true,
        ncyclesperiteration=ncpi,
        maxsize=complexity
    )
    for i in 0:nsamples-1
        n = scale * growth^i

        data = @btime case(input, n)

        trees, complexity = calculateSR(data, 30, options)

        push!(plotdata, (trees=trees, complexity=complexity, label="N=$n"))
    end

    data = @btime case(input, ntest)

    return plotSampleDep(plotdata, data, options)
end

pSample1_losses = SampleRun(case1)
savefig(pSample1_losses, "Images/sample_losses_case1.pdf")

pSample2_losses = SampleRun(case2)
savefig(pSample2_losses, "Images/sample_losses_case2.pdf")

exit()

# Noise Robust Test

data1 = case(var1, 100)
data2 = case(var2, 100)
#data3 = case3(var3, 100)

inv(x) = 1 / x
options = Options(
    binary_operators=(+, *, ^),
    unary_operators=(inv, -),
    batching=true,
    ncyclesperiteration=ncpi,
    maxsize=complexity
)

data1_1 = deepcopy(data1)
data1_10 = deepcopy(data1)
noise = randn(length(data1.y))
data1_1.y .= (1.0 .+ 0.01 .* noise .* data1_1.y) .* data1_1.y
data1_10.y .= (1.0 .+ 0.1 .* noise .* data1_10.y) .* data1_10.y

data2_1 = deepcopy(data2)
data2_10 = deepcopy(data2)
noise = randn(length(data2.y))
data2_1.y .= (1.0 .+ 0.01 .* noise .* data2_1.y) .* data2_1.y
data2_10.y .= (1.0 .+ 0.1 .* noise .* data2_10.y) .* data2_10.y

#data3_1 = deepcopy(data3)
#data3_10 = deepcopy(data3)
#noise = randn(length(data3.y))
#data3_1.y .= (1.0 .+ 0.01 .* noise .* data3_1.y) .* data3_1.y
#data3_10.y .= (1.0 .+ 0.1 .* noise .* data3_10.y) .* data3_10.y

trees1, complexity1 = calculateSR(data1, 1000, options)
trees2, complexity2 = calculateSR(data2, 1000, options)
#trees3, complexity3 = calculateSR(data3, 1000, options)
trees1_1, complexity1_1 = calculateSR(data1_1, 1000, options)
trees2_1, complexity2_1 = calculateSR(data2_1, 1000, options)
#trees3_1, complexity3_1 = calculateSR(data3_1, 1000, options)
trees1_10, complexity1_10 = calculateSR(data1_10, 1000, options)
trees2_10, complexity2_10 = calculateSR(data2_10, 1000, options)
#trees3_10, complexity3_10 = calculateSR(data3_10, 1000, options)

sets1 = [(data=data1, trees=trees1, complexity=complexity1, label="Exact"), (data=data1_1, trees=trees1_1, complexity=complexity1_1, label="1% Noise"), (data=data1_10, trees=trees1_10, complexity=complexity1_10, label="10% Noise")]
sets2 = [(data=data2, trees=trees2, complexity=complexity2, label="Exact"), (data=data2_1, trees=trees2_1, complexity=complexity2_1, label="1% Noise"), (data=data2_10, trees=trees2_10, complexity=complexity2_10, label="10% Noise")]
#sets3 = [(data=data3, trees=trees3, complexity=complexity3, label="Exact"), (data=data3_1, trees=trees3_1, complexity=complexity3_1, label="1% Noise"), (data=data3_10, trees=trees3_10, complexity=complexity3_10, label="10% Noise")]


function plotcase(sets)
    losses = []
    plot_font = "Computer Modern"
    default(fontfamily=plot_font)
    plt = plot(title="Loss over Complexity", dp=1000)
    for set in sets
        values = [eval_tree_array(tree, set.data.X, options)[1] for tree in set.trees]
        residuals = values
        for residual in residuals
            residual .= residual - first(sets).data.y
        end
        loss = log10.([sum(abs2, residual) / length(residual) for residual in residuals])
        push!(losses, loss)
        plot!(set.complexity, loss, label=set.label, xlabel="Complexity", ylabel="Loss")
    end
    return plt
end

p1_losses = plotcase(sets1)
p2_losses = plotcase(sets2)
#p3_losses = plotcase(sets3)

savefig(p1_losses, "Images/losses_case1.pdf")
savefig(p2_losses, "Images/losses_case2.pdf")
#savefig(p3_losses, "Images/losses_case3.pdf")

open("result.txt", "a") do io
    println(io, "Case 1:")
    for set in sets1
        println(io, node_to_symbolic(set.trees[end], options))
    end
    println(io)

    println(io, "Case 2:")
    for set in sets2
        println(io, node_to_symbolic(set.trees[end], options))
    end
    println(io)

    #println("Case 3:")
    #for set in sets3
    #    println(io, node_to_symbolic(set.trees[end], options))
    #end
end