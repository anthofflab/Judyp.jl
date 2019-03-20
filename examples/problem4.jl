using Distributions

function getproblem4()    
    # Set the economic parameters

    ex_params = (
        η = 2.,        # Consumption smoothing parameter
        beta = 0.8,     # discount factor
        κ = 0.25,       # Capital share
        δ = 0.1,       # Capital depreciation rate
        A_0 = 1.4,      # Initial level of technology
        k_0 = 0.8,   # Initial level of effective capital K_0/A_0
        ψ = 1,
        B = 0.621301775,
        ρ = 0.78,
        σ = 0.0067
    )
    
    problem = DynProgProblem()
       
    set_transition_function!(problem) do s, s_new, x, up, p
        choices = x

        s_new[1] = (1-p.δ)*s[1] + choices[2]

        s_new[2] = s[2]^p.ρ * exp(up[1])
    end

    set_payoff_function!(problem) do s, x, p
        choices = x

        ret = zero(eltype(x))
        consumption = choices[1]
        hours = choices[3]
        ret += (consumption^(1-p.η))/(1-p.η) - p.B * hours^(1 + p.ψ) / (1 + p.ψ)

        return ret
    end

    set_discountfactor!(problem, ex_params.beta)

    set_constraints_function!(problem) do s, x, g, p
        choices = x

        k = s[1]
        Y = s[2] * p.A_0 * (k^p.κ) * choices[3]^(1-p.κ)
        g[1] = Y - choices[1] - choices[2]
    end

    set_exogenous_parameters!(problem, ex_params)
    
    add_state_variable!(problem, :k, 1., 0.7,1.3, 10)
    add_state_variable!(problem, :A, 1., 0.8853, 1.1295, 10)

    add_choice_variable!(problem, :C, 0., 100000000000., 0.1)
    add_choice_variable!(problem, :I, 0., 100000000000., 0.1)
    add_choice_variable!(problem, :z, 0., 100000000000., 0.1)

    add_constraint!(problem, 0., Inf, false)

    ## adds Gauss-Hermite weights and nodes for N = 8
    set_uncertain_weights!(problem, [1.99604072e-04, 1.70779830e-02, 2.07802326e-01, 6.61147013e-01,6.61147013e-01, 2.07802326e-01, 1.70779830e-02, 1.99604072e-04] ./ sqrt(pi))

    add_uncertain_parameter!(problem, Normal(0., ex_params.σ), [-2.93063742, -1.98165676, -1.15719371, -0.38118699,  0.38118699, 1.15719371,  1.98165676,  2.93063742] .* sqrt(2) .* ex_params.σ)
    

	return problem
end
