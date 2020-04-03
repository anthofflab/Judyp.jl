## Example 6: energy model with two capital stocks, denoted by k and q; 
## - investment decisions in k and q are made one period in advance
## - Cobb-Douglas production function with labor and a CES composite of k and q, with EoS given by σ

function getproblem6()    
    # Set the economic parameters

    ex_params = (
        β = 0.970225,       # discount factor
        α = 0.3,
        γ = 0.04,
        ψ = -1.0,
        ξ = 0.25,
        L = 0.531882,
        A_2100 = 37.791077477,      # initial level of technology
        δ_c = 0.16,    # capital depreciation rate
        δ_ws = 0.168236,
        δ_st = 0.16,
        δ_h = 0.0559391,
        δ_n = 0.07388127187120652,
        g = 0.02,
        ζ_ws = 1.4292699785,
        ζ_st = 2.75,
        ζ_h = 5.598,
        ζ_n = 4.555,
        h = 17.52,
        κ = 0.372,
        Ω = 2.9281161957,
        k_n = 0.012074,
        k_h = 0.011234
    )

    problem = DynProgProblem()
       
    set_transition_function!(problem) do s, s_new, x, up, p
        s_new[1] = ( (1-p.δ_c)*s[1] + x[2] ) / exp(p.g)
        s_new[2] = ( (1-p.δ_ws)*s[2] + x[3] / p.ζ_ws ) / exp(p.g)
    end

    set_payoff_function!(problem) do s, x, p
        ret = zero(eltype(x))
        consumption = x[1]
        ret += log(consumption)
        return ret
    end

    set_discountfactor!(problem, ex_params.β)

    set_constraints_function!(problem) do s, x, g, p           
        k_c = s[1]
        k_ws = s[2] 
        k_st = p.κ * k_ws / p.Ω
        i_st = p.κ * p.ζ_st / p.Ω * (x[3] - (p.δ_ws - p.δ_st) * k_ws)
        e = p.h * (k_st + 0.9 * p.k_n / p.A_2100 + 0.33 * p.k_h / p.A_2100)             
        y = k_c^p.α * p.L^(1 - p.α - p.γ) * e^p.γ * p.ξ^(p.γ/p.ψ)
        g[1] = y - x[1] - x[2] - x[3] - i_st - p.δ_n * p.ζ_n * p.k_n / p.A_2100 - p.δ_h * p.ζ_h * p.k_h / p.A_2100
    end

    set_exogenous_parameters!(problem, ex_params)
    
    add_state_variable!(problem, :k_c, 0.702189, 0.7 * 0.702189 , 2 * 0.702189, 10)
    add_state_variable!(problem, :k_ws, 0.009989, 0.009989, 30 * 0.009989, 10)

    add_choice_variable!(problem, :c, 0., 100000000000., 0.5)
    add_choice_variable!(problem, :i_c, 0., 100000000000., 0.1)
    add_choice_variable!(problem, :i_ws, 0., 100000000000., 0.1)

    add_constraint!(problem, 0., Inf, false)

	return problem
end
