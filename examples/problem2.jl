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

    function transition(k,k_new, x, up, p)
        choices = reshape(x,(2,p.regions))

        for i=1:length(k)
            k_new[i] = (1-p.δ)*k[i] + choices[2,i]
        end
    end

    function payoff(s, x, p)
        choices = reshape(x,(2,p.regions))

        ret = zero(eltype(x))
        for i=1:p.regions
            consumption = choices[1,i]
            ret += (consumption^(1-p.η))/(1-p.η)
        end
        return ret
    end

    function discountfactor(state, p)
        return exp(-p.ρ)
    end

    function constraints(state, x, g, p)
        choices = reshape(x,(2,p.regions))

        for i=1:p.regions
            g[i] = Y(state[i], p) - choices[1,i] - choices[2,i]
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
    g_uncertain_weights = Float64[]
    g_uncertain_nodes = Matrix{Float64}(undef,0,0)

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
        g_uncertain_weights,
        g_uncertain_nodes,
        ex_params)

	return problem
end
