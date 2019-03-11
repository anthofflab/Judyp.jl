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
    
       
    function transition(k,k_new, x, unc_p, p)
        choices = x

        k_new[1] = (1-p.δ)*k[1] + choices[2]

        k_new[2] = k[2]^p.rho * exp(unc_p[1])
    end

    function payoff(s, x, p)
        choices = x

        ret = zero(eltype(x))
        consumption = choices[1]
        hours = choices[3]
        ret += (consumption^(1-p.η))/(1-p.η) - p.B * hours^(1 + p.psi) / (1 + p.psi)

        return ret
    end

    function discountfactor(state, p)
        return p.beta
    end

    function constraints(state, x, g, p)
        choices = x

        k = state[1]
        Y = state[2] * p.A_0 * (k^p.κ) * choices[3]^(1-p.κ)
        g[1] = Y - choices[1] - choices[2]

    end

    num_node = [10, 10]
    s_min = [0.7, 0.887]
    s_max = [1.3, 1.128]
    x_min = ones(3) * 0.0
    x_max = ones(3) * 100000000000.
    x_init = ones(3) .* 0.1
    g_min = [0.0]
    g_max = [Inf]
    g_linear = repeat([false], inner=[1])

    uncertain_weights = [0.2, 0.4, 0.2]
    uncertain_nodes = reshape([1.,1.,1.], 3, :)

	problem = DynProgProblem(
		transition,
		payoff,
        constraints,
        discountfactor,
		num_node,
		s_min,
		s_max,
		x_min,
		x_max,
        x_init,
        g_min,
        g_max,
        g_linear,
        uncertain_weights,
        uncertain_nodes,
        ex_params)

	return problem
end
