mutable struct LinearGlobalFunctionApproximator{W <: AbstractVector{Float64}} <: GlobalFunctionApproximator
    weights::W
end

function fit!(lgfa::LinearGlobalFunctionApproximator, dataset_input::AbstractMatrix{Float64},
              dataset_output::AbstractVector{Float64})
    # TODO: Should we assume bias=false by default and delegate to user in feature vector? like [1 x x^2 ...]
    lgfa.weights = llsq(dataset_input,dataset_output,bias=false)
end


function compute_value(lgfa::LinearGlobalFunctionApproximator, v::V) where V <: AbstractVector{Float64}
    @assert length(lgfa.weights) == length(v)
    return dot(lgfa.weights, v)
end

function compute_value(lgfa::LinearGlobalFunctionApproximator, 
                        v_list::AbstractVector{V}) where V <: AbstractVector{Float64}
    @assert length(v_list) > 0
    vals = [compute_value(lgfa,v) for v in v_list]
    return vals
end