function getproblem2(regions=2)
    # Set the economic parameters

    ex_params = (
        regions = regions,
        η = 2.,        # Consumption smoothing parameter
        ρ = 0.015,     # Annual rate of pure time preference
        κ = 0.3,       # Capital share
        δ = 0.1,       # Capital depreciation rate
        g = 0.0,       # Rate of technological progress
        L = 7.,        # Global labor force in billions of people
        K_0 = 80.,       # Initial capital stock
        A_0 = 1.,        # Initial level of technology
        k_0 = 80. / 1.   # Initial level of effective capital K_0/A_0
    )

    Y(k, p) = (k^p.κ)*p.L^(1-p.κ)

    problem = DynProgProblem()

    set_transition_function!(problem) do s, s_new, x, up, p
        choices = reshape(x,(2,p.regions))

        for i=1:length(s)
            s_new[i] = (1-p.δ)*s[i] + choices[2,i]
        end
    end

    set_payoff_function!(problem) do s, x, p
        choices = reshape(x,(2,p.regions))

        ret = zero(eltype(x))
        for i=1:p.regions
            consumption = choices[1,i]
            ret += (consumption^(1-p.η))/(1-p.η)
        end
        return ret
    end

    set_discountfactor!(problem, 1-ex_params.ρ)

    set_constraints_function!(problem) do s, x, g, p
        choices = reshape(x,(2,p.regions))

        for i=1:p.regions
            g[i] = Y(s[i], p) - choices[1,i] - choices[2,i]
        end
    end

    set_exogenous_parameters!(problem, ex_params)

    for i=1:regions
        add_state_variable!(problem, 10., 1., 100., 10)
        add_choice_variable!(problem, 0., 100000000000., 0.1)
        add_constraint!(problem, 0., Inf, false)
    end

	return problem
end
