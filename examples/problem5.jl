## Example 5: NGM with two capital stocks, denoted by k and q; 
## - investment decisions in k and q are made one period in advance
## - Cobb-Douglas production function with labor and a CES composite of k and q, with EoS given by σ

function getproblem5()    
    # Set the economic parameters

    ex_params = (
        η = 2.,        # consumption smoothing parameter
        ψ = 1.,        # labor parameter
        B = 1.,        # disutility weight for labor
        β = 0.9,       # discount factor
        κ = 0.25,      # capital share in production function
        ω = 0.5,       # CES distribution parameter
        σ = 0.8,       # elasticity of substitution
        δ = 0.1,       # capital depreciation rate
        A_0 = 1.,      # initial level of technology
    )
    
    problem = DynProgProblem()
       
    set_transition_function!(problem) do s, s_new, x, up, p
        s_new[1] = (1-p.δ)*s[1] + x[2]
        s_new[2] = (1-p.δ)*s[2] + x[3]
    end

    set_payoff_function!(problem) do s, x, p
        ret = zero(eltype(x))
        consumption = x[1]
        ret += (consumption^(1-p.η))/(1-p.η) - p.B * x[4]^(1 + p.ψ) / (1 + p.ψ)
        return ret
    end

    set_discountfactor!(problem, ex_params.β)

    set_constraints_function!(problem) do s, x, g, p           
        k = s[1]
        q = s[2]
        h = x[4]
        Y = p.A_0 * ( p.ω * k^((p.σ-1)/p.σ) + (1 - p.ω) * q^((p.σ-1)/p.σ)  )^(p.σ * p.κ/(p.σ-1)) * h^(1-p.κ)
        g[1] = Y - x[1] - x[2] - x[3]
    end

    set_exogenous_parameters!(problem, ex_params)
    
    add_state_variable!(problem, :K, 0.4, 0.3, 0.8, 10)
    add_state_variable!(problem, :Q, 0.4, 0.3, 0.8, 10)

    add_choice_variable!(problem, :C, 0., 100000000000., 0.5)
    add_choice_variable!(problem, :Ik, 0., 100000000000., 0.1)
    add_choice_variable!(problem, :Iq, 0., 100000000000., 0.1)
    add_choice_variable!(problem, :H, 0., 100000000000., 1.)

    add_constraint!(problem, 0., Inf, false)

	return problem
end
