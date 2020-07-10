using Judyp
include("problem1.jl")

for i = 1:4
problem = getproblem1(i)

@time res = solve(problem, print_level=0)
end
    
# simres = simulate(res, 50, 0)
