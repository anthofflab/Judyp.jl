function getproblem4()    
    # Set the economic parameters

    ex_params = (
        η = 2.,        # Consumption smoothing parameter
        beta = 0.8,     # discount factor
        κ = 0.25,       # Capital share
        δ = 0.1,       # Capital depreciation rate
        A_0 = 1.4,      # Initial level of technology
        k_0 = 0.8,   # Initial level of effective capital K_0/A_0
        psi = 1,
        B = 0.621301775,
        rho = 0.78
    )
    
    problem = DynProgProblem()
       
    set_transition_function!(problem) do s, s_new, x, up, p
        choices = x

        s_new[1] = (1-p.δ)*s[1] + choices[2]

        s_new[2] = s[2]^p.rho * exp(up[1])
    end

    set_payoff_function!(problem) do s, x, p
        choices = x

        ret = zero(eltype(x))
        consumption = choices[1]
        hours = choices[3]
        ret += (consumption^(1-p.η))/(1-p.η) - p.B * hours^(1 + p.psi) / (1 + p.psi)

        return ret
    end

    set_discountfactor!(problem, ex_params.beta)

    set_constraints_function!(problem) do s, x, g, p
        choices = x

        k = s[1]
        Y = s[2] * p.A_0 * (k^p.κ) * choices[3]^(1-p.κ)
        g[1] = Y - choices[1] - choices[2]
    end

    add_state_variable!(problem, 1., 0.7,1.3, 10)
    add_state_variable!(problem, 1., 0.887,1.128, 10)

    add_choice_variable!(problem, 0., 100000000000., 0.1)
    add_choice_variable!(problem, 0., 100000000000., 0.1)
    add_choice_variable!(problem, 0., 100000000000., 0.1)

    add_constraint!(problem, 0., Inf, false)

    set_uncertain_weights!(problem, [0.2, 0.4, 0.2])

    add_uncertain_parameter!(problem, [1.,1.,1.])

	return problem
end
