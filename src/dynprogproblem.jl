mutable struct DynProgProblem
    transition_function::Union{Function,Nothing}
    payoff_function::Union{Function,Nothing}
    constraint_function::Function
    discountfactor_function::Union{Function,Nothing}

    num_node::Array{Int,1}

    s0::Vector{Float64}
    s_min::Array{Float64,1}
    s_max::Array{Float64,1}
    s_name::Vector{Symbol}

    x_min::Array{Float64,1}
    x_max::Array{Float64,1}
    x_init::Array{Float64,1}
    x_name::Vector{Symbol}

    g_min::Array{Float64,1}
    g_max::Array{Float64,1}
    g_linear::Array{Bool,1}

    uncertain_weights::Vector{Float64}
    uncertain_nodes::Matrix{Float64}
    uncertain_distributions::Vector{Distribution}

    ex_params::Union{NamedTuple,Nothing}

    function DynProgProblem()
        return new(
            nothing,
            nothing,
            (state, x, g, p) -> nothing,
            nothing,
            Vector{Int}(undef,0),
            Vector{Float64}(undef,0),
            Vector{Float64}(undef,0),
            Vector{Float64}(undef,0),
            Vector{Symbol}(undef, 0),
            Vector{Float64}(undef,0),
            Vector{Float64}(undef,0),
            Vector{Float64}(undef,0),
            Vector{Symbol}(undef, 0),
            Vector{Float64}(undef,0),
            Vector{Float64}(undef,0),
            Vector{Bool}(undef,0),
            Vector{Float64}(undef,0),
            Matrix{Float64}(undef,0,0),
            Vector{Distribution}(undef, 0),
            nothing
        )
    end
end

function Base.show(io::IO, p::DynProgProblem)
    println(io, "Dynamic programming problem with")
    println(io, "  $(length(p.num_node)) state variables")
    for i=1:length(p.num_node)
        println(io, "    State $i: $(p.num_node[i]) nodes over [$(p.s_min[i]), $(p.s_max[i])]")
    end
    println(io, "  $(length(p.x_min)) choice variables")
    for i=1:length(p.x_min)
        println(io, "    Choice $i: bounds [$(p.x_min[i]), $(p.x_max[i])] with initial value $(p.x_init[i])")
    end
    println(io, "  $(length(p.g_min)) constraints")
    for i=1:length(p.g_min)
        println(io, "    Constraint $i: bounds [$(p.g_min[i]), $(p.g_max[i])] $(p.g_linear[i] ? "linear" : "")")
    end
    println(io, "  $(size(p.uncertain_nodes,2)) uncertain parameters")
end

function set_transition_function!(p::DynProgProblem, f::Function)
    p.transition_function = f
    nothing
end
set_transition_function!(f::Function, p::DynProgProblem) = set_transition_function!(p, f)

function set_payoff_function!(p::DynProgProblem, f::Function)
    p.payoff_function = f
    nothing
end
set_payoff_function!(f::Function, p::DynProgProblem) = set_payoff_function!(p, f)

function set_constraints_function!(p::DynProgProblem, f::Function)
    p.constraint_function = f
    nothing
end
set_constraints_function!(f::Function, p::DynProgProblem) = set_constraints_function!(p, f)

function set_discountfactor_function!(p::DynProgProblem, f::Function)
    p.discountfactor_function = f
    nothing
end
set_discountfactor_function!(f::Function, p::DynProgProblem) = set_discountfactor_function!(p, f)

function set_discountfactor!(p::DynProgProblem, discountfactor::Float64)
    set_discountfactor_function!(p) do state, p
        return discountfactor
    end
    nothing
end

function add_state_variable!(p::DynProgProblem, name::Symbol, s0::Float64, s_min::Float64, s_max::Float64, nodes::Int)
    push!(p.s0, s0)
    push!(p.s_min, s_min)
    push!(p.s_max, s_max)
    push!(p.num_node, nodes)    
    push!(p.s_name, name)
    nothing
end

function add_choice_variable!(p::DynProgProblem, name::Symbol, x_min::Float64, x_max::Float64, x_init::Float64=mean([x_min, x_max]))
    push!(p.x_min, x_min)
    push!(p.x_max, x_max)
    push!(p.x_init, x_init)
    push!(p.x_name, name)
    nothing
end

function add_constraint!(p::DynProgProblem, g_min::Float64, g_max::Float64, g_linear::Bool=false)
    push!(p.g_min, g_min)
    push!(p.g_max, g_max)
    push!(p.g_linear, g_linear)
    nothing
end

function set_uncertain_weights!(p::DynProgProblem, uncertain_weights::Vector{Float64})
    p.uncertain_weights = uncertain_weights
end

function add_uncertain_parameter!(p::DynProgProblem, dist::Distribution, uncertain_nodes::Vector{Float64})
    length(p.uncertain_weights) == 0 && error("You first need to set the uncertain weights with `set_uncertain_weights!`.")
    length(p.uncertain_weights) != length(uncertain_nodes) && error("You need to provide the same number of nodes as weights.")
    size(p.uncertain_nodes,1) > 0 && error("Currently only one uncertain parameter is supported.")


    p.uncertain_nodes = size(p.uncertain_nodes,1) > 0 ? hcat(p.uncertain_nodes, reshape(uncertain_nodes, length(uncertain_nodes), 1)) : reshape(uncertain_nodes, length(uncertain_nodes), 1)
    push!(p.uncertain_distributions, dist)
    nothing
end

function set_exogenous_parameters!(p::DynProgProblem, ex_params)
    p.ex_params = ex_params
    nothing
end
