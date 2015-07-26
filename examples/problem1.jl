function getproblem1(regions=2)
    # Set the economic parameters

    η = 2.        # Consumption smoothing parameter
    ρ = 0.015     # Annual rate of pure time preference
    κ = 0.3       # Capital share
    δ = 0.1       # Capital depreciation rate
    g = 0.0       # Rate of technological progress
    L = 7.        # Global labor force in billions of people

    K_0 = 80.       # Initial capital stock
    A_0 = 1.        # Initial level of technology
    k_0 = K_0/A_0   # Initial level of effective capital

    Y(k) = (k^κ)*L^(1-κ)

    function transition(k,k_new, x)
        for i=1:length(k)
            k_new[i] = (1-δ)*k[i] + x[i] * Y(k[i])
        end
    end

    function payoff(s, x)
        ret = zero(eltype(x))
        for i=1:length(x)
            consumption = (1-x[i]) * Y(s[i])
            ret += (consumption^(1-η))/(1-η)
        end
        return ret
    end

    function discountfactor(state)
        return exp(-ρ)
    end

    function constraints(state, x, g)
        return
    end

    num_node = ones(Int, regions) .* 10
    s_min = ones(regions) .* 1.0
    s_max = ones(regions) .* 100.0
    x_min = ones(regions) .* 0.0
    x_max = ones(regions) .* 1.0
    x_init = ones(regions) .* 0.5
    g_min = Array(Float64, 0)
    g_max = Array(Float64, 0)
    g_linear = Array(Bool, 0)

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
        g_linear)

	return problem
end
