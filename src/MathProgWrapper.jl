using ForwardDiff
using MathProgBase
using MathProgBase.SolverInterface

type JuDPNLPEvaluator <: AbstractNLPEvaluator
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
            ForwardDiff.gradient(f, mutates=true), #Float64, fadtype=:dual, n=x_len),
            g,
            ForwardDiff.jacobian(g, mutates=true, output_length=g_len), #Float64, fadtype=:dual, n=x_len, m=g_len),
            x_len,
            g_len,
            g_linear,
            debug_trace,
            Array(Float64, g_len, x_len))
    end
end

function MathProgBase.SolverInterface.initialize(d::JuDPNLPEvaluator, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

function MathProgBase.SolverInterface.features_available(d::JuDPNLPEvaluator)
    return [:Grad, :Jac, :Hess]
end

function MathProgBase.SolverInterface.eval_f(d::JuDPNLPEvaluator, x)
    d.debug_trace && print("f($x) ->")
    y = d.f(x)
    d.debug_trace && println(" $y")
    return y
end

function MathProgBase.SolverInterface.eval_g(d::JuDPNLPEvaluator, g, x)
    d.debug_trace && print("g($x) ->")
    d.g(g, x)
    d.debug_trace && println(" $g")
end

function MathProgBase.SolverInterface.eval_grad_f(d::JuDPNLPEvaluator, grad_f, x)
    d.debug_trace && print("f'($x) ->")
    try
        d.grad_f(grad_f, x)
    catch e
        if isa(e, DomainError)
            grad_f[:] = NaN
        else
            rethrow()
        end
    end
    d.debug_trace && println(" $grad_f")
end

function MathProgBase.SolverInterface.jac_structure(d::JuDPNLPEvaluator)
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

function MathProgBase.SolverInterface.hesslag_structure(d::JuDPNLPEvaluator)
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

function MathProgBase.SolverInterface.eval_jac_g(d::JuDPNLPEvaluator, J, x)
    d.debug_trace && print("g'($x) ->")
    try
        d.jac_g(d.temp_jac_g_output, x)
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

function MathProgBase.SolverInterface.eval_hesslag(d::JuDPNLPEvaluator, H, x, σ, μ)
    error("Not yet implemented")
end

function MathProgBase.SolverInterface.isobjlinear(d::JuDPNLPEvaluator)
    return false
end

function MathProgBase.SolverInterface.isobjquadratic(d::JuDPNLPEvaluator)
    return false
end

function MathProgBase.SolverInterface.isconstrlinear(d::JuDPNLPEvaluator, i::Int)
    return d.g_linear[i]
end
