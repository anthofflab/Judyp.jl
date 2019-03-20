struct SimResult
    solution::DynProgSolution
    state_vars::Array{Float64,3}
    choice_vars::Array{Float64,3}
end

function Base.show(io::IO, p::SimResult)
    print(io, "Judyp simulation result for $(size(p.state_vars, 3)-1) Monte Carlo runs and $(size(p.state_vars,2)) time periods.")
end

function simulate(sol::DynProgSolution, timeperiods, n)

    state_vars = Array{Float64,3}(undef,length(sol.problem.s_min), timeperiods, n + 1)
    choice_vars = Array{Float64,3}(undef,length(sol.problem.x_min), timeperiods, n + 1)
    new_state = Vector{Float64}(undef, length(sol.problem.s_min))
    uncertain_parameters = Vector{Float64}(undef, length(sol.problem.uncertain_distributions))

    for run_id in 1:(n+1)

        state_vars[:,1,run_id] .= sol.problem.s0

        for t=1:timeperiods
            curr_state = view(state_vars, :, t, run_id)
            curr_policy = sol.policyfun(curr_state)

            choice_vars[:,t,run_id] .= curr_policy

            if t<timeperiods
                if run_id==1
                    uncertain_parameters[:] .= mode.(sol.problem.uncertain_distributions)
                else
                    uncertain_parameters[:] .= rand.(sol.problem.uncertain_distributions)                    
                end
                sol.problem.transition_function(curr_state, new_state, curr_policy, uncertain_parameters, sol.problem.ex_params)

                state_vars[:,t+1,run_id] .= new_state        
            end
        end
    end

    return SimResult(sol, state_vars, choice_vars)
end

function get_table(sm::SimResult)
    var_names = Vector{Symbol}(undef,0)
    run_ids = Vector{Int}(undef,0)
    timesteps = Vector{Int}(undef,0)
    values = Vector{Float64}(undef,0)

    for run_id in 1:size(sm.state_vars, 3), timestep in 1:size(sm.state_vars, 2), var_id in 1:size(sm.state_vars, 1)
        push!(var_names, sm.solution.problem.s_name[var_id])
        push!(run_ids, run_id-1)
        push!(timesteps, timestep)
        push!(values, sm.state_vars[var_id, timestep, run_id])
    end

    for run_id in 1:size(sm.choice_vars, 3), timestep in 1:size(sm.choice_vars, 2), var_id in 1:size(sm.choice_vars, 1)
        push!(var_names, sm.solution.problem.x_name[var_id])
        push!(run_ids, run_id-1)
        push!(timesteps, timestep)
        push!(values, sm.choice_vars[var_id, timestep, run_id])
    end

    return DataTable(variable=var_names, runid=run_ids, timestep=timesteps, value=values)
end
