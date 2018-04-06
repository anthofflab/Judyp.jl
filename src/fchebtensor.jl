using ForwardDiff

mutable struct valuefunstate
  n_nodes::Array{Int64,1}
  s_min::Array{Float64,1}
  s_max::Array{Float64,1}

  temp_curr_node::Array{Int64,1}
  temp_cheb_vals_float64::Array{Array{Float64,1},1}
  temp_cheb_vals_duals::Array{Array{ForwardDiff.Dual{Void,Float64,4},1},1}
end

function cheb_nodes(num_node, a, b)
    [(a+b)/2 + (b-a)/2 * cos((num_node - i + 0.5)/num_node * π) for i in 1:num_node]
end

function genvaluefunstate(n_nodes, s_min, s_max)
  vs = valuefunstate(n_nodes, s_min,
    s_max,
    Array{Int}(length(n_nodes)),
    Array{Vector{Float64}}(length(n_nodes)),
    Array{Vector{ForwardDiff.ForwardDiff.Dual{Void,Float64,4}}}(length(n_nodes))
    )

  for i=1:length(n_nodes)
    vs.temp_cheb_vals_float64[i] = Array{Float64}(n_nodes[i])
    vs.temp_cheb_vals_duals[i] = Array{ForwardDiff.Dual{Void,Float64,4}}(n_nodes[i])
  end

  return vs
end

function getrighttemparray(::Type{Float64},state::valuefunstate)
  return state.temp_cheb_vals_float64
end

function getrighttemparray(::Type{ForwardDiff.Dual{Void,Float64,4}},state::valuefunstate)
  return state.temp_cheb_vals_duals
end

function valuefun(s_unscaled::Array{T,1}, c::Array{Float64,1}, state::valuefunstate) where {T <: Number}
  n_states = length(state.n_nodes)
  temparray = getrighttemparray(T,state)

  # Phase 1
  # Evaluate all the chebyshev basis functions and store the results in temparray
  for l=1:n_states
    curr_temparray = temparray[l]

    # Scale into -1 to 1 range
    z = 2(s_unscaled[l]-state.s_min[l])/(state.s_max[l]-state.s_min[l])-1

    for k=1:state.n_nodes[l]
      if k==1
        # Compute degree 0 chebyshev polynomial
        curr_temparray[k] = 1.
      elseif k==2
        # Compute degree 1 chebyshev polynomial
        curr_temparray[k] = z
      else
        # Compute degree n>1 chebyshev polynomial
        curr_temparray[k] = 2z * curr_temparray[k-1] - curr_temparray[k-2]
      end
    end
  end

  # Phase 2
  # Tensor basis related computations
  ret = zero(T)

  state.temp_curr_node[:] = 1
  curr_s_pos = 1
  clen = length(c)
  for i=1:clen
    ϕ = zero(T) + c[i]
    for l=1:n_states
      ϕ *= temparray[l][state.temp_curr_node[l]]
    end

    # Counting
    state.temp_curr_node[curr_s_pos] += 1
    if state.temp_curr_node[curr_s_pos] > state.n_nodes[curr_s_pos] && i!=clen
      while state.temp_curr_node[curr_s_pos] > state.n_nodes[curr_s_pos]
        state.temp_curr_node[curr_s_pos] = 1
        curr_s_pos += 1
        state.temp_curr_node[curr_s_pos] += 1
      end
      curr_s_pos = 1
    end

    ret = ret + ϕ
  end
  return ret
end
