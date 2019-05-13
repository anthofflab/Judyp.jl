## Example 3: NGM with energy as third production factor
## - energy is produced using labor, with one unit of energy costing 1/ρ units of labor

function getproblem3()    
    # Set the economic parameters

    ex_params = (
        η = 2.,        # consumption smoothing parameter
        ψ = 1.,        # labor parameter
        B = 1.,        # disutility weight for labor
        β = 0.9,       # discount factor
        κ = 0.25,      # capital share in production function
        γ = 0.05,      # energy share in production function
        δ = 0.1,       # capital depreciation rate
        ρ = 0.9,       # inverse of price of energy
        A_0 = 1.,      # initial level of technology
#         k_0 = 0.8,     # initial level of effective capital K_0/A_0
    )
    
    problem = DynProgProblem()
       
    set_transition_function!(problem) do s, s_new, x, up, p
        s_new[1] = (1-p.δ)*s[1] + x[2]
    end

    set_payoff_function!(problem) do s, x, p
        ret = zero(eltype(x))
        consumption = x[1]
        ret += (consumption^(1-p.η))/(1-p.η) - p.B * (x[3] + x[4])^(1 + p.ψ) / (1 + p.ψ)
        return ret
    end

    set_discountfactor!(problem, ex_params.β)

    set_constraints_function!(problem) do s, x, g, p           
        k = s[1]
        h = x[3]
        e = p.ρ * x[4]
        Y = p.A_0 * (k^p.κ) * e^p.γ * h^(1-p.κ-p.γ)
        g[1] = Y - x[1] - x[2]
    end

    set_exogenous_parameters!(problem, ex_params)
    
    add_state_variable!(problem, :k, 1., 0.7, 1.3, 10)

    add_choice_variable!(problem, :C, 0., 100000000000., 1.)
    add_choice_variable!(problem, :I, 0., 100000000000., 0.1)
    add_choice_variable!(problem, :H, 0., 100000000000., 0.5)
    add_choice_variable!(problem, :He, 0., 100000000000., 0.1)

    add_constraint!(problem, 0., Inf, false)

	return problem
end
