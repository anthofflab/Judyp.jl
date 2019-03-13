using Distributed

addprocs()

using Judyp
@everywhere using NLopt
@everywhere using Ipopt

include("problem1.jl")

problem = getproblem1()

nlopt_solver = () -> NLoptSolver(algorithm=:LD_SLSQP)
ipopt_solver = () -> IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0.)

@time res = psolve(problem, solver_constructors=[nlopt_solver, ipopt_solver], print_level=2)

simres = simulate(res, 50)
