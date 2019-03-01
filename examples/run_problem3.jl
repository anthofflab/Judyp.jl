# using Distributed
# addprocs(4)

using Judyp
# @everywhere using NLopt
# @everywhere using Ipopt
# using KNITRO
# using Gadfly
# cd("examples")
include("problem3.jl")

problem = getproblem3()

# nlopt_solver = () -> NLoptSolver(algorithm=:LD_SLSQP)
# solver=IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0., linear_solver="ma57")
# ipopt_solver = () -> IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0.)
# solver=KnitroSolver(hessopt=KTR_HESSOPT_FINITE_DIFF, KTR_PARAM_OUTLEV=0, KTR_PARAM_HONORBNDS=KTR_HONORBNDS_ALWAYS, KTR_PARAM_PAR_CONCURRENT_EVALS=KTR_PAR_CONCURRENT_EVALS_NO, KTR_PARAM_BAR_FEASIBLE=KTR_BAR_FEASIBLE_GET_STAY)

# addprocs()

@time res = solve(problem, print_level=2)

# v = res.valuefun

# data = [v([i,l]) for i=0.:1.:100., l=0.:1.:100.]

# plot([x->v([x,10.]), x->v([10.,x])], 0., 100.)
