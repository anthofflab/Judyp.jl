using ForwardDiff
using MathProgBase
import MathProgBase.MathProgSolverInterface

type JuDPNLPEvaluator <: MathProgSolverInterface.AbstractNLPEvaluator
    f::Function
    grad_f::Function
    g::Function
    jac_g::Function

    xlen::Int
    glen::Int

    g_linear::Array{Bool,1}

    debug_trace::Bool

    temp_jac_g_output::Array{Float64,2}

    function JuDPNLPEvaluator(f, g, x_len, g_linear;debug_trace=false)
        g_len = length(g_linear)
        new(
            f,
            forwarddiff_gradient!(f, Float64, fadtype=:dual, n=x_len),
            g,
            forwarddiff_jacobian!(g, Float64, fadtype=:dual, n=x_len, m=g_len),
            x_len,
            g_len,
            g_linear,
            debug_trace,
            Array(Float64, g_len, x_len))
    end
end

function MathProgSolverInterface.initialize(d::JuDPNLPEvaluator, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

function MathProgSolverInterface.features_available(d::JuDPNLPEvaluator)
    return [:Grad, :Jac, :Hess]
end

function MathProgSolverInterface.eval_f(d::JuDPNLPEvaluator, x)
    d.debug_trace && print("f($x) ->")
    y = d.f(x)
    d.debug_trace && println(" $y")
    return y
end

function MathProgSolverInterface.eval_g(d::JuDPNLPEvaluator, g, x)
    d.debug_trace && print("g($x) ->")
    d.g(x, g)
    d.debug_trace && println(" $g")
end

function MathProgSolverInterface.eval_grad_f(d::JuDPNLPEvaluator, grad_f, x)
    d.debug_trace && print("f'($x) ->")
    try
        d.grad_f(x, grad_f)
    catch e
        if isa(e, DomainError)
            grad_f[:] = NaN
        else
            retrhow()
        end
    end
    d.debug_trace && println(" $grad_f")
end

function MathProgSolverInterface.jac_structure(d::JuDPNLPEvaluator)
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

function MathProgSolverInterface.hesslag_structure(d::JuDPNLPEvaluator)
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

function MathProgSolverInterface.eval_jac_g(d::JuDPNLPEvaluator, J, x)
    d.debug_trace && print("g'($x) ->")
    try
        d.jac_g(x, d.temp_jac_g_output)
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

function MathProgSolverInterface.eval_hesslag(d::JuDPNLPEvaluator, H, x, σ, μ)
    error("Not yet implemented")
end

function MathProgSolverInterface.isobjlinear(d::JuDPNLPEvaluator)
    return false
end

function MathProgSolverInterface.isobjquadratic(d::JuDPNLPEvaluator)
    return false
end

function MathProgSolverInterface.isconstrlinear(d::JuDPNLPEvaluator, i::Int)
    return d.g_linear[i]
end
