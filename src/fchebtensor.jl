using ForwardDiff
using Iterators

mutable struct ValueFunState{NCHOICE}
  n_nodes::Array{Int64,1}
  s_min::Array{Float64,1}
  s_max::Array{Float64,1}

  temp_curr_node::Array{Int64,2}
  temp_cheb_vals_float64::Array{Float64,2}
  temp_cheb_vals_duals::Array{ForwardDiff.Dual{Void,Float64,NCHOICE},2}
end

function cheb_nodes(num_node, a, b)
    [(a+b)/2 + (b-a)/2 * cos((num_node - i + 0.5)/num_node * π) for i in 1:num_node]
end

function genvaluefunstate(n_nodes, s_min, s_max, nchoice)
  

  vs = ValueFunState{nchoice}(n_nodes, s_min,
    s_max,
    Array{Int}(prod(n_nodes),length(n_nodes)),
    #Pick Max nodes
    Array{Float64,2}(length(n_nodes),maximum(n_nodes)),
    Array{ForwardDiff.ForwardDiff.Dual{Void,Float64,nchoice},2}(length(n_nodes),maximum(n_nodes))
    )

    #Tensor Product 
    arrs = [1:i for i in n_nodes]
    k=1
    for v in product(arrs...)
        i = 1
        for el in v
            vs.temp_curr_node[k, i] = el
            i += 1
        end
        k += 1
    end

  return vs
end

function getrighttemparray(::Type{Float64},state::ValueFunState)
  return state.temp_cheb_vals_float64
end

function getrighttemparray(::Type{ForwardDiff.Dual{Void,Float64,NCHOICE}},state::ValueFunState) where {NCHOICE}
  return state.temp_cheb_vals_duals
end




function valuefun(s_unscaled::Array{T,1}, c::Array{Float64,1}, state::ValueFunState) where {T <: Number}
  n_states = length(state.n_nodes)
  temparray = getrighttemparray(T,state)

  # Phase 1
  # Evaluate all the chebyshev basis functions and store the results in temparray
  for l=1:n_states

    # Scale into -1 to 1 range
    z = 2(s_unscaled[l]-state.s_min[l])/(state.s_max[l]-state.s_min[l])-1

    
    if(state.n_nodes[l]>=2)
        # Compute degree 1 chebyshev polynomial
        temparray[l,2] = z
        # Compute degree 0 chebyshev polynomial
        temparray[l,1] = 1.

    elseif(state.n_nodes[l]>=1)
        # Compute degree 0 chebyshev polynomial
        temparray[l,1] = 1.
    end


    for k=3:state.n_nodes[l]
        temparray[l,k] = 2z * temparray[l,k-1] - temparray[l,k-2]
    end
  end


  # Phase 2
  

  ret = zero(T)
  
  curr_s_pos = 1
  
  
  clen = length(c)
  for i=1:clen
    ϕ = zero(T) + c[i]
    for l=1:n_states
      ϕ *= temparray[l,state.temp_curr_node[curr_s_pos,l]] 
      
    end
    curr_s_pos += 1

    ret = ret + ϕ
  end
  return ret
end

