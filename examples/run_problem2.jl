using Judyp
include("problem2.jl")

problem = getproblem2()

@time res = solve(problem, print_level=2)

simres = simulate(res, 50)
