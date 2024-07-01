module HTSR

import SymbolicRegression as SR

function calculateSR(X::Matrix{Float64},y::Vector{Float64},iter::Integer,opt)
    hallOfFame = SR.equation_search(X, y, niterations=iter, options=opt, parallelism=:multithreading);
    dominating = SR.calculate_pareto_frontier(X, y, hallOfFame, opt);

    trees = [member.tree for member in dominating];

    return trees
end

end