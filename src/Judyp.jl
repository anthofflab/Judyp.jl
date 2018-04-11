module Judyp

export DynProgProblem

export solve

using MathProgBase
using CompEcon
using ProgressMeter

mutable struct DynProgProblem{T}
    transition_function::Function
    payoff_function::Function
    constraint_function::Function
    discountfactor_function::Function

    num_node::Array{Int,1}

    s_min::Array{Float64,1}
    s_max::Array{Float64,1}

    x_min::Array{Float64,1}
    x_max::Array{Float64,1}
    x_init::Array{Float64,1}

    g_min::Array{Float64,1}
    g_max::Array{Float64,1}
    g_linear::Array{Bool,1}

    ex_params::T
end

mutable struct DynProgState{T}
    problem::DynProgProblem{T}

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

    Φ::Vector{Matrix{Float64}}

    mathprog_problems

    function DynProgState(problem::DynProgProblem{T}, solver_constructors) where {T}
        clen = prod(problem.num_node)
        c = zeros(clen)
        c_old = zeros(clen)
        x_initial_value = Array{Float64}(length(problem.x_min))
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

        Φ = Array{Array{Float64,2}}(length(problem.num_node))
        for l=length(problem.num_node):-1:1
          Φ_per_state = [cos((problem.num_node[l]-i+.5)*(j-1)*π/problem.num_node[l]) for i=1:problem.num_node[l],j=1:problem.num_node[l]]
          Φ[length(problem.num_node) - l + 1] = Φ_per_state
        end               

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

        return new{T}(
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
            Φ,
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
    elapsed_solver
    it
    diagnostics::JudypDiagnostics
end

include("fchebtensor.jl")
include("foptimize.jl")
include("MathProgWrapper.jl")

function solve(problem::DynProgProblem;
        solver_constructors::Vector{Function}=[()->IpoptSolver(hessian_approximation="limited-memory", print_level=0)],
        print_level=1,
        maxit=10000,
        tol=1e-3)

    diag = JudypDiagnostics()

    dpstate = DynProgState(problem, solver_constructors)      

    elapsed_solver = Array{Float64}(0)
    for it=1:maxit
        progress = ProgressMeter.Progress(dpstate.clen, "Iteration $it...")#, "Iteration $it...", 50)

        dpstate.c, dpstate.c_old = dpstate.c_old, dpstate.c
        dpstate.opt_state.c = dpstate.c_old

        dpstate.new_v, dpstate.old_v = dpstate.old_v, dpstate.new_v

        node_range = CartesianRange(tuple(problem.num_node...))
        for (i, curr_node)=enumerate(node_range)
            for l=1:length(problem.num_node)
                dpstate.opt_state.s_curr_state[l] = dpstate.s_nodes[l][curr_node.I[l]]
            end

            for l=1:length(problem.x_max)
                dpstate.x_initial_value[l] = dpstate.x_init[i,l]
            end

            solution_found = false
            for mp in dpstate.mathprog_problems
                setwarmstart!(mp,dpstate.x_initial_value)
                elapsed = @elapsed optimize!(mp)
                push!(elapsed_solver, elapsed)
                stat = status(mp)

                if stat==:Optimal || stat==:FeasibleApproximate
                    solution_found = true

                    dpstate.x_init[i, :] = getsolution(mp)

                    dpstate.new_v[i] = - getobjval(mp)

                    break
                end
                diag.count_first_solver_failed = diag.count_first_solver_failed + 1
            end

            if !solution_found
                error("Didn't converge at iteration $it and node $i")
            end

            if print_level>=2
                ProgressMeter.next!(progress)
            end
        end
        dpstate.c[:] = BasisMatrices.ckronxi(dpstate.Φ, dpstate.new_v)

        #step1 = norm(c .- c_old,Inf)
        step1 = maximum(abs, dpstate.c .- dpstate.c_old)
        step2 = norm(dpstate.new_v .- dpstate.old_v,Inf)

        if step1<tol  #converged
            if print_level>=1
                println("Function iteration converged after $it iterations with max. coefficient difference of $step1")
            end
            return DynProgSolution(dpstate.c, q->valuefun(q, dpstate.opt_state.c, dpstate.opt_state.value_fun_state), elapsed_solver, it, diag)
        end

        if print_level>=2
            println("Function iteration $it: max coeff change $step1, max value change $step2")
        end

    end
    error("Function iteration with reached the maximum number of iterations")
end

end # module
