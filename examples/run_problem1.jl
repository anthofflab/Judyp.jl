using Judyp
include("problem1.jl")

problem = getproblem1()

@time res = solve(problem, print_level=0)
    
simres = simulate(res, 50, 0)
