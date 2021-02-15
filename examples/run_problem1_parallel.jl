using Distributed

addprocs()

@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)

using Judyp
@everywhere using NLopt
@everywhere using Ipopt

include("problem1.jl")

for i = 2:3
    problem = getproblem1(i)

    nlopt_solver = () -> NLoptSolver(algorithm=:LD_SLSQP)
    ipopt_solver = () -> IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0.)

    @time res = psolve(problem, solver_constructors=[nlopt_solver, ipopt_solver], print_level=0)
end

# simres = simulate(res, 50, 0)
