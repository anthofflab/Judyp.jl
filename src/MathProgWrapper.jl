using ForwardDiff
using MathProgBase
using MathProgBase.SolverInterface

type JudypNLPEvaluator <: AbstractNLPEvaluator
    f::Function
    g::Function

    xlen::Int
    glen::Int

    g_linear::Array{Bool,1}

    debug_trace::Bool

    temp_jac_g_output::Array{Float64,2}
    temp_g_output::Array{Float64,1}

    function JudypNLPEvaluator(f, g, x_len, g_linear;debug_trace=false)
        g_len = length(g_linear)
        new(
            f,
            g,
            x_len,
            g_len,
            g_linear,
            debug_trace,
            Array(Float64, g_len, x_len),
            Array(Float64, g_len))
    end
end

function MathProgBase.SolverInterface.initialize(d::JudypNLPEvaluator, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

function MathProgBase.SolverInterface.features_available(d::JudypNLPEvaluator)
    return [:Grad, :Jac, :Hess]
end

function MathProgBase.SolverInterface.eval_f(d::JudypNLPEvaluator, x)
    d.debug_trace && print("f($x) ->")
    y = d.f(x)
    d.debug_trace && println(" $y")
    return y
end

function MathProgBase.SolverInterface.eval_g(d::JudypNLPEvaluator, g, x)
    d.debug_trace && print("g($x) ->")
    d.g(g, x)
    d.debug_trace && println(" $g")
end

function MathProgBase.SolverInterface.eval_grad_f(d::JudypNLPEvaluator, grad_f, x)
    d.debug_trace && print("f'($x) ->")
    try
        ForwardDiff.gradient!(grad_f, d.f, x)
    catch e
        if isa(e, DomainError)
            grad_f[:] = NaN
        else
            rethrow()
        end
    end
    d.debug_trace && println(" $grad_f")
end

function MathProgBase.SolverInterface.jac_structure(d::JudypNLPEvaluator)
    rows = Array(Int,0)
    cols = Array(Int,0)

    for r in 1:d.glen
        for c in 1:d.xlen
            push!(rows,r)
            push!(cols,c)
        end
    end

    return rows, cols
end

function MathProgBase.SolverInterface.hesslag_structure(d::JudypNLPEvaluator)
    rows = Array(Int,0)
    cols = Array(Int,0)

    for r in 1:d.xlen
        for c in 1:d.xlen
            push!(rows,r)
            push!(cols,c)
        end
    end

    return rows, cols
end

function MathProgBase.SolverInterface.eval_jac_g(d::JudypNLPEvaluator, J, x)
    d.debug_trace && print("g'($x) ->")
    try
        ForwardDiff.jacobian!(d.temp_jac_g_output, d.g, d.temp_g_output, x)
    catch e
        if isa(e, DomainError)
            d.temp_jac_g_output[:,:] = NaN
        else
            rethrow()
        end
    end
    d.debug_trace && println(" $(d.temp_jac_g_output)")
    j = 1
    for r in 1:d.glen
        for c in 1:d.xlen
            J[j] = d.temp_jac_g_output[r,c]
            j += 1
        end
    end
end

function MathProgBase.SolverInterface.eval_hesslag(d::JudypNLPEvaluator, H, x, σ, μ)
    error("Not yet implemented")
end

function MathProgBase.SolverInterface.isobjlinear(d::JudypNLPEvaluator)
    return false
end

function MathProgBase.SolverInterface.isobjquadratic(d::JudypNLPEvaluator)
    return false
end

function MathProgBase.SolverInterface.isconstrlinear(d::JudypNLPEvaluator, i::Int)
    return d.g_linear[i]
end
