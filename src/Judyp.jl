module Judyp

using Distributed, LinearAlgebra, Statistics

export DynProgProblem, set_transition_function!, set_payoff_function!,
    set_constraints_function!, set_discountfactor_function!, add_state_variable!,
    add_choice_variable!, add_constraint!, set_uncertain_weights!,
    add_uncertain_parameter!, set_exogenous_parameters!, set_discountfactor!

export solve, psolve, simulate, get_table

using MathProgBase
using CompEcon
using ProgressMeter
using ParallelDataTransfer
using NLopt
using Ipopt
using Distributions
using DataTables

include("dynprogproblem.jl")
include("ntarraywrapper.jl")


mutable struct DynProgState{T}
    problem::DynProgProblem

    clen::Int
    c::Vector{Float64}
    c_old::Vector{Float64}
    x_initial_value::Vector{Float64}
    x_init::Matrix{Float64}
    new_v::Vector{Float64}
    old_v::Vector{Float64}

    s_nodes::Vector{Vector{Float64}}

    value_fun_state
    opt_state

    # Φ::Vector{Matrix{Float64}}

    mathprog_problems

    function DynProgState(problem::DynProgProblem, solver_constructors)
        clen = prod(problem.num_node)
        c = zeros(clen)
        c_old = zeros(clen)
        x_initial_value = Array{Float64}(undef, length(problem.x_min))
        x_init = ones(length(c), length(problem.x_min))
        for i=1:clen
            for l=1:length(problem.x_min)
                x_init[i,l] = problem.x_init[l]
            end
        end
        new_v = zeros(length(c))
        old_v = zeros(length(c))

        s_nodes = [cheb_nodes(problem.num_node[i], problem.s_min[i], problem.s_max[i]) for i=1:length(problem.num_node)]

        value_fun_state = genvaluefunstate(problem.num_node,problem.s_min,problem.s_max, length(problem.x_init))

        opt_state = genoptimizestate(problem, value_fun_state)

                       

        evaluator = JudypNLPEvaluator(
            x -> -objective_f(x, opt_state),
            (g,x) -> constraint_f(x, g, opt_state),
            length(problem.x_min),
            problem.g_linear,
            debug_trace=false
        )
      
        # MathProg stuff
        mathprog_problems = [NonlinearModel(sc()) for sc in solver_constructors]
      
        for mp in mathprog_problems
            loadproblem!(
                mp,
                length(problem.x_min),
                length(problem.g_min),
                problem.x_min, problem.x_max,
                problem.g_min, problem.g_max,
                :Min,
                evaluator)
        end        

        return new{typeof(problem.ex_params)}(
            problem,
            clen,
            c,
            c_old,
            x_initial_value,
            x_init,
            new_v,
            old_v,
            s_nodes,
            value_fun_state,
            opt_state,
            mathprog_problems
        )
    end
end

mutable struct JudypDiagnostics
    count_first_solver_failed::Int64

    function JudypDiagnostics()
        return new(0)
    end
end

mutable struct DynProgSolution
    c
    valuefun
    policyfun
    elapsed_solver
    it
    diagnostics::JudypDiagnostics
    problem::DynProgProblem
end

function Base.show(io::IO, s::DynProgSolution)
    print(io, "Dynamic programming solution")
end

include("fchebtensor.jl")
include("foptimize.jl")
include("MathProgWrapper.jl")

function solve_node(dpstate::DynProgState, diag, i, curr_node, it)
    for l=1:length(dpstate.problem.num_node)
        dpstate.opt_state.s_curr_state[l] = dpstate.s_nodes[l][curr_node.I[l]]
    end

    for l=1:length(dpstate.problem.x_max)
        dpstate.x_initial_value[l] = dpstate.x_init[i,l]
    end

    solution_found = false
    new_v = 0.
    for mp in dpstate.mathprog_problems
        setwarmstart!(mp,dpstate.x_initial_value)
        elapsed = @elapsed optimize!(mp)
        # push!(elapsed_solver, elapsed)
        stat = status(mp)

        if stat==:Optimal || stat==:FeasibleApproximate
            solution_found = true

            dpstate.x_init[i, :] = getsolution(mp)

            new_v = - getobjval(mp)

            break
        end
        diag.count_first_solver_failed = diag.count_first_solver_failed + 1
    end

    if !solution_found
        error("Didn't converge at iteration $it and node $i")
    end

    return new_v
end

function pnodeloopinner(dpstate, diag, node_range, batch_size, it)
    vs = zeros(length(Iterators.take(Iterators.drop(enumerate(node_range), (myid()-2) * batch_size), batch_size)))
    for (li, (i, curr_node))=enumerate(Iterators.take(Iterators.drop(enumerate(node_range), (myid()-2) * batch_size), batch_size))
        v_new = Judyp.solve_node(dpstate, diag, i, curr_node, it)

        vs[li] = v_new

        # if print_level>=2
        #     ProgressMeter.next!(progress)
        # end
    end   
    return vs
end

function pnodeloop()
    return Judyp.pnodeloopinner(Main.dpstate, Main.diag, Main.node_range, Main.batch_size, Main.it)
end

function mypassobj(target::AbstractVector{Int}, nm::Symbol, value; to_mod=Main)
    r = RemoteChannel(myid())
    put!(r, value)
    @sync for to in target
        @spawnat(to, Core.eval(to_mod, Expr(:(=), nm, fetch(r))))
    end
    nothing
end

function psolve(problem::DynProgProblem;
        solver_constructors=nothing,#[()->IpoptSolver(hessian_approximation="limited-memory", print_level=0)],
        print_level=1,
        maxit=10000,
        tol=1e-3)

    if solver_constructors===nothing
        solver_constructors = [() -> NLoptSolver(algorithm=:LD_SLSQP, maxtime=2), () -> IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0.)]
    end

    @eval @everywhere using Judyp

    println("A")
    mypassobj(workers(), :problem, problem)
    println("B")
    # @passobj 1 workers() solver_constructors
    @broadcast solver_constructors = [() -> NLoptSolver(algorithm=:LD_SLSQP, maxtime=2), () -> IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0.)]
    println("C")
    @broadcast dpstate = Judyp.DynProgState(problem, solver_constructors)
    println("D")
    @broadcast node_range = CartesianIndices(tuple(problem.num_node...))
    println("E")
    @broadcast batch_size = Int(round(length(node_range)/length(workers()), RoundUp))
    println("F")
    @broadcast diag = Judyp.JudypDiagnostics()

    clen = prod(problem.num_node)

    c = zeros(clen)
    c_old = zeros(clen)

    new_v = zeros(clen)
    old_v = zeros(clen)

    Φ = Array{Array{Float64,2}}(undef, length(problem.num_node))
        for l=length(problem.num_node):-1:1
          Φ_per_state = [cos((problem.num_node[l]-i+.5)*(j-1)*π/problem.num_node[l]) for i=1:problem.num_node[l],j=1:problem.num_node[l]]
          Φ[length(problem.num_node) - l + 1] = Φ_per_state
    end

    elapsed_solver = Array{Float64}(undef, 0)
    for it=1:maxit
        println("Iteration $it...")
        # progress = ProgressMeter.Progress(dpstate.clen, "Iteration $it...")

        c, c_old = c_old, c

        mypassobj(workers(), :it, it)

        mypassobj(workers(), :c, c_old)
        # @broadcast dpstate.c = c
        @broadcast dpstate.opt_state.c = c

        new_v, old_v = old_v, new_v

        @broadcast new_vs = Judyp.pnodeloop()

        vs = asyncmap(i->getfrom(i, :new_vs), workers())

        # @broadcast for (i, curr_node)=Iterators.take(Iterators.drop(enumerate(node_range), (myid()-2) * batch_size), batch_size)
        #     v_new = Judyp.solve_node(dpstate, i, curr_node)

        #     dpstate.new_v[i] = v_new

        #     # if print_level>=2
        #     #     ProgressMeter.next!(progress)
        #     # end
        # end
        for (i,v) in enumerate(Iterators.flatten(vs))
            new_v[i] = v
        end

        c[:] = BasisMatrices.ckronxi(Φ, new_v)
        
        #step1 = norm(c .- c_old,Inf)
        step1 = maximum(abs, c .- c_old)
        step2 = norm(new_v .- old_v,Inf)

        if step1<tol  #converged
            if print_level>=1
                println("Function iteration converged after $it iterations with max. coefficient difference of $step1")
            end

            dpstate = DynProgState(problem, solver_constructors)
            dpstate.opt_state.c = c
            dpstate.c = c

            p_func = s -> begin
                dpstate.opt_state.s_curr_state[:] .= s            
            
                for mp in dpstate.mathprog_problems
                    setwarmstart!(mp,problem.x_init)
                    optimize!(mp)

                    stat = status(mp)
            
                    if stat==:Optimal || stat==:FeasibleApproximate
                        solution_found = true
            
                         return getsolution(mp)
                    end
                end
            
                error("Couldn't solve")
            end
       
            return DynProgSolution(dpstate.c, q->valuefun(q, dpstate.opt_state.c, dpstate.opt_state.value_fun_state), p_func, elapsed_solver, it, JudypDiagnostics(), problem)
        end

        if print_level>=2
            println("Function iteration $it: max coeff change $step1, max value change $step2")
        end

    end
    error("Function iteration with reached the maximum number of iterations")
end

function solve(problem::DynProgProblem;
    solver_constructors::Vector{Function}=[() -> NLoptSolver(algorithm=:LD_SLSQP, maxtime=2,ftol_rel=1e-12), () -> IpoptSolver(hessian_approximation="limited-memory", max_iter=10000, print_level=0, bound_relax_factor=0.)],
    print_level=1,
    maxit=10000,
    tol=1e-3)

    diag = JudypDiagnostics()

    dpstate = DynProgState(problem, solver_constructors)     
    
    Φ = Array{Array{Float64,2}}(undef, length(problem.num_node))
    for l=length(problem.num_node):-1:1
        Φ_per_state = [cos((problem.num_node[l]-i+.5)*(j-1)*π/problem.num_node[l]) for i=1:problem.num_node[l],j=1:problem.num_node[l]]
        Φ[length(problem.num_node) - l + 1] = Φ_per_state
    end

    elapsed_solver = Array{Float64}(undef, 0)
    for it=1:maxit
        progress = ProgressMeter.Progress(dpstate.clen, "Iteration $it...")

        dpstate.c, dpstate.c_old = dpstate.c_old, dpstate.c
        dpstate.opt_state.c = dpstate.c_old

        dpstate.new_v, dpstate.old_v = dpstate.old_v, dpstate.new_v

        node_range = CartesianIndices(tuple(problem.num_node...))
        for (i, curr_node)=enumerate(node_range)
            v_new = solve_node(dpstate, diag, i, curr_node, it)

            dpstate.new_v[i] = v_new

            if print_level>=2
                ProgressMeter.next!(progress)
            end
        end
        dpstate.c[:] = BasisMatrices.ckronxi(Φ, dpstate.new_v)

        #step1 = norm(c .- c_old,Inf)
        step1 = maximum(abs, dpstate.c .- dpstate.c_old)
        step2 = norm(dpstate.new_v .- dpstate.old_v,Inf)

        if step1<tol  #converged
            if print_level>=1
                println("Function iteration converged after $it iterations with max. coefficient difference of $step1")
            end

            p_func = (s, init_guess) -> begin
                dpstate.opt_state.s_curr_state[:] .= s            
            
                for mp in dpstate.mathprog_problems
                    setwarmstart!(mp,init_guess)
                    optimize!(mp)

                    stat = try
                        status(mp)
                    catch err
                        :FAILURE
                    end
            
                    if stat==:Optimal || stat==:FeasibleApproximate
                        solution_found = true
            
                        return getsolution(mp)
                    end
                end
            
                error("Couldn't solve")
            end

            return DynProgSolution(dpstate.c, q->valuefun(q, dpstate.opt_state.c, dpstate.opt_state.value_fun_state), p_func, elapsed_solver, it, diag, problem)
        end

        if print_level>=2
            println("Function iteration $it: max coeff change $step1, max value change $step2")
        end

    end
    error("Function iteration with reached the maximum number of iterations")
end

include("sim.jl")

end # module
