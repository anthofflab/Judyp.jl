using ForwardDiff

mutable struct OptimizeState{NCHOICE,T}
    n_states::Int64
    f_transition::Function
    f_payoff::Function
    f_constraint::Function
    f_discountfactor::Function
    s_curr_state::Array{Float64,1}
    c::Array{Float64,1}
    value_fun_state::ValueFunState
    temp_new_state_float64::Array{Float64,1}
    temp_new_state_dual::Array{ForwardDiff.Dual{Nothing,Float64,NCHOICE},1}
    ex_params::T
    uncertain_weights::Vector{Float64}
    uncertain_nodes::Matrix{Float64}
end

function genoptimizestate(problem::DynProgProblem, value_fun_state)
    nchoice = length(problem.x_init)
    os = OptimizeState{nchoice,typeof(problem.ex_params)}(
        length(value_fun_state.n_nodes),
        problem.transition_function,
        problem.payoff_function,
        problem.constraint_function,
        problem.discountfactor_function,
        Array{Float64}(undef, length(value_fun_state.n_nodes)),
        zeros(1), # This could be left uninitialized
        value_fun_state,
        Array{Float64}(undef, length(value_fun_state.n_nodes)),
        Array{ForwardDiff.Dual{Nothing,Float64,nchoice}}(undef, length(value_fun_state.n_nodes)),
        problem.ex_params,
        problem.uncertain_weights,
        problem.uncertain_nodes)
    return os
end

function getrighttemparray(::Type{Float64},state::OptimizeState)
    return state.temp_new_state_float64
end

function getrighttemparray(::Type{ForwardDiff.Dual{Nothing,Float64,NCHOICE}},state::OptimizeState) where NCHOICE
    return state.temp_new_state_dual
end

function objective_f(x::Array{T,1}, state::OptimizeState) where {T <: Number}
    k_new = getrighttemparray(T, state)

    payoff = state.f_payoff(state.s_curr_state, x, state.ex_params)
    
    v = zero(T)

    if length(state.uncertain_weights) == 0
        state.f_transition(state.s_curr_state, k_new, x, nothing, state.ex_params)
        v += valuefun(k_new, state.c, state.value_fun_state)
    else
        for i=1:length(state.uncertain_weights)
            state.f_transition(state.s_curr_state, k_new, x, view(state.uncertain_nodes, i, :), state.ex_params)
            v += state.uncertain_weights[i] * valuefun(k_new, state.c, state.value_fun_state)
        end
    end

    return payoff + state.f_discountfactor(state.s_curr_state, state.ex_params) * v
end

function constraint_f(x::Array{T,1}, g, state::OptimizeState) where {T <: Number}
    state.f_constraint(state.s_curr_state, x, g, state.ex_params)
end
