using NLopt
using Ipopt

include("dynprog.jl")
include("problem1.jl")

function perftest()
    problem = getproblem1()
    solvers = [
        NLoptSolver(algorithm=:LD_MMA),
        NLoptSolver(algorithm=:LD_SLSQP),
        NLoptSolver(algorithm=:LD_LBFGS),
        NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART),
        NLoptSolver(algorithm=:LD_VAR2),
        NLoptSolver(algorithm=:LD_VAR1),
        IpoptSolver(hessian_approximation="limited-memory", print_level=0, linear_solver="ma57")]#,
        #KnitroSolver(hessopt=KTR_HESSOPT_FINITE_DIFF, KTR_PARAM_OUTLEV=0)]

    for i in solvers
        println("Now doing $i")
        solve(problem, solver=i, print_level=0)
        @time res1, vfun, el, it = solve(problem, solvers=[i], print_level=0)
        println("Number of solve calls: ", length(el))
        println("Mean solve time in seconds: ", mean(el))
        #println("f evals:    ", sprintf1("%'d", f_eval_count))
        #println("grad evals: ", sprintf1("%'d", f_grad_eval_count/2)) # Divide by two because grad eval calls this twice currently
        println("Iteration count: $it")
        println("")
    end
end

perftest()
