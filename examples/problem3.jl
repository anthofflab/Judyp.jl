function getproblem3()
    regions=2
    # Set the economic parameters

    η = 2.        # Consumption smoothing parameter
    ρ = 0.015     # Annual rate of pure time preference
    κ = 0.3       # Capital share
    δ = 0.1       # Capital depreciation rate
    g = 0.0       # Rate of technological progress
    L = [2., 5.]        # Global labor force in billions of people

    K_0 = 80.       # Initial capital stock
    A_0 = 1.        # Initial level of technology
    k_0 = K_0/A_0   # Initial level of effective capital

    function transition(k,k_new, x)
        choices = reshape(x,(2,regions))

        for i=1:length(k)
            k_new[i] = (1-δ)*k[i] + choices[2,i]
        end
    end

    function payoff(s, x)
        choices = reshape(x,(2,regions))

        ret = zero(eltype(x))
        for i=1:regions
            consumption = choices[1,i]
            ret += (consumption^(1-η))/(1-η)
        end
        return ret
    end

    function discountfactor(state)
        return exp(-ρ)
    end

    function constraints(state, x, g)
        choices = reshape(x,(2,regions))

        for i=1:regions
            k = state[i]
            Y = (k^κ)*L[i]^(1-κ)
            g[i] = Y - choices[1,i] - choices[2,i]
        end
    end

    num_node = ones(Int, regions) .* 10
    s_min = ones(regions) .* 1.0
    s_max = ones(regions) .* 100.0
    x_min = ones(regions*2) .* 0.0
    x_max = ones(regions*2) .* 100000000000.
    x_init = ones(regions*2) .* 0.1
    g_min = ones(regions) .* 0.0
    g_max = ones(regions) .* Inf
    g_linear = repeat([false], inner=[regions])

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
