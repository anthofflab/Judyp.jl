using NLopt
using Ipopt
include("dynprog.jl")
include("problem1.jl")

warmstart_problem = getproblem1(1)
problem = getproblem1()

solver = NLoptSolver(algorithm=:LD_SLSQP)
#solver=IpoptSolver(hessian_approximation="limited-memory", print_level=0, bound_relax_factor=0.)

solve(warmstart_problem, solvers=[solver], print_level=0);

@time res = solve(problem, solvers=[solver], print_level=1)
