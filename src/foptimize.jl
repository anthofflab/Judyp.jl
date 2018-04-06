using ForwardDiff

mutable struct optimizestate{NCHOICE}
    n_states::Int64
    f_transition::Function
    f_payoff::Function
    f_constraint::Function
    f_discountfactor::Function
    s_curr_state::Array{Float64,1}
    c::Array{Float64,1}
    value_fun_state::valuefunstate
    temp_new_state_float64::Array{Float64,1}
    temp_new_state_dual::Array{ForwardDiff.Dual{Void,Float64,NCHOICE},1}
end

function genoptimizestate(problem::DynProgProblem, value_fun_state)
    nchoice = length(problem.x_init)
    os = optimizestate{nchoice}(
        length(value_fun_state.n_nodes),
        problem.transition_function,
        problem.payoff_function,
        problem.constraint_function,
        problem.discountfactor_function,
        Array{Float64}(length(value_fun_state.n_nodes)),
        zeros(1), # This could be left uninitialized
        value_fun_state,
        Array{Float64}(length(value_fun_state.n_nodes)),
        Array{ForwardDiff.Dual{Void,Float64,nchoice}}(length(value_fun_state.n_nodes)))
    return os
end

function getrighttemparray(::Type{Float64},state::optimizestate)
    return state.temp_new_state_float64
end

function getrighttemparray(::Type{ForwardDiff.Dual{Void,Float64,NCHOICE}},state::optimizestate) where NCHOICE
    return state.temp_new_state_dual
end

function objective_f(x::Array{T,1}, state::optimizestate) where {T <: Number}
    k_new = getrighttemparray(T, state)

    state.f_transition(state.s_curr_state, k_new, x)

    payoff = state.f_payoff(state.s_curr_state, x)

    v = valuefun(k_new, state.c, state.value_fun_state)

    return payoff + state.f_discountfactor(state.s_curr_state) * v
end

function constraint_f(x::Array{T,1}, g, state::optimizestate) where {T <: Number}
    state.f_constraint(state.s_curr_state, x, g)
end
