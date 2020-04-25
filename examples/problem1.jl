function getproblem1(regions=2)
    # Set the economic parameters

    ex_params = (
        η = 2.,        # Consumption smoothing parameter
        ρ = 0.015,     # Annual rate of pure time preference
        κ = 0.3,       # Capital share
        δ = 0.1,       # Capital depreciation rate
        g = 0.0,       # Rate of technological progress
        L = 7.,        # Global labor force in billions of people

        K_0 = 80.,       # Initial capital stock
        A_0 = 1.,        # Initial level of technology
        k_0 = 80. / 1.   # Initial level of effective capital
    )

    Y(k, p) = (k^p.κ)*p.L^(1-p.κ)

    problem = DynProgProblem()

    set_transition_function!(problem) do k, k_new, x, up, p
        for i=1:length(k)
            k_new[i] = (1-p.δ)*k[i] + x[i] * Y(k[i], p)
        end
    end

    set_payoff_function!(problem) do s, x, p
        ret = zero(eltype(x))
        for i=1:length(x)
            consumption = (1-x[i]) * Y(s[i], p)
            ret += (consumption^(1-p.η))/(1-p.η)
        end
        return ret
    end

    set_discountfactor!(problem, exp(-ex_params.ρ))
    
    set_exogenous_parameters!(problem, ex_params)

    for i=1:regions
        add_state_variable!(problem, Symbol("Capital_"*string(i)), 10., 1., 100., 10)
        add_choice_variable!(problem, Symbol("Savings_"*string(i)), 0., 1.)
    end

	return problem
end
