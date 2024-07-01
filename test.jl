cd(@__DIR__);
println(pwd());

using Pkg;
Pkg.activate(pwd());

include("HTSR.jl");

using SymbolicRegression
using .HTSR

n = 100
X = Array{Float64,2}(undef, 2, n^2)
Ra = @view X[1, :]
Pr = @view X[2, :]

Pr_ref = collect(range(0.5, 10.0, n))
Ra_ref = collect(range(1.0e2, 1.0e8, n))
for i in 1:n
    for j in 1:n
        Pr[(i-1)*n+j] = Pr_ref[i]
        Ra[(i-1)*n+j] = Ra_ref[j]
    end
end

fc(Pr, Ra) = 0.677 * ((20.0 / (21.0 * Pr)) + 1.0)^(-0.25) * Ra^(0.25)

y = fc.(Pr, Ra)

inv(x) = 1/x

options = Options(
    binary_operators=(+, *, ^),
    unary_operators=(inv,-),
    batching=true,
    ncyclesperiteration=1000,
    maxsize=20
)

trees = HTSR.calculateSR(X, y, 1000, options)

using Plots

points = exp10.(range(0.0f0, 10.0f0, 1000))
Pr_test = 1.5

f(Ra) = fc(Pr_test, Ra)

tree = trees[end]
Y = Array{Float64,2}(undef, 2, length(points))
Y[1, :] .= points
Y[2, :] .= Pr_test
result, worked = eval_tree_array(tree, Y, options)

if worked
    plot(log.(points), [log.(abs.(f.(points))) log.(abs.(result))])
else
    println("NaN or Infinity detected.")
end