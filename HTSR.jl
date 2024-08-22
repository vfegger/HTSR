module HTSR

import SymbolicRegression as SR

export Data, calculateSR

mutable struct Data{T}
    X::Matrix{T}
    y::Vector{T}
end

function Data(X::Matrix{T},y::Vector{T}) where T <: Number
    return Data{T}(X,y)
end

function calculateSR(data::Data{T},iterations::Integer,options) where T <: Number
    hallOfFame = SR.equation_search(data.X, data.y, niterations=iterations, options=options, parallelism=:multithreading);
    dominating = SR.calculate_pareto_frontier(data.X, data.y, hallOfFame, options);

    trees = [member.tree for member in dominating];
    complexity = [SR.compute_complexity(member,options) for member in dominating]
    return trees, complexity
end

end