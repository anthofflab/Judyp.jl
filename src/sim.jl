struct SimResult
    state_vars::Vector{Matrix{Float64}}
    choice_vars::Vector{Matrix{Float64}}
end

function simulate(sol::DynProgSolution, timeperiods)

    state_vars = [Matrix{Float64}(undef, 1, timeperiods) for i in 1:length(sol.problem.s_min)]
    choice_vars = [Matrix{Float64}(undef, 1, timeperiods) for i in 1:length(sol.problem.x_min)]

    run_id = 1

    for i=1:length(state_vars)
        state_vars[i][run_id,1] = sol.problem.s0[i]
    end

    for t=1:timeperiods
        curr_state = [state_vars[i][run_id,t] for i=1:length(state_vars)]
        curr_policy = sol.policyfun(curr_state)

        for i=1:length(choice_vars)
            choice_vars[i][run_id,t] = curr_policy[i]
        end

        if t<timeperiods
            new_state = similar(curr_state)

            sol.problem.transition_function(curr_state, new_state, curr_policy, [], sol.problem.ex_params)
    
            for i=1:length(state_vars)
                state_vars[i][run_id,t+1] = new_state[i]
            end
        end
    end

    return SimResult(state_vars, choice_vars)
end
