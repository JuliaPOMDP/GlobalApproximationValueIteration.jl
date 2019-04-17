mutable struct LinearGlobalFunctionApproximator{W <: AbstractArray} <: GlobalFunctionApproximator
    weights::W
end

function fit!(lgfa::LinearGlobalFunctionApproximator, dataset_input::AbstractMatrix{T},
              dataset_output::AbstractArray{T}) where T
    # TODO: Should we assume bias=false by default and delegate to user in feature vector? like [1 x x^2 ...]
    lgfa.weights = llsq(dataset_input, dataset_output, bias=false)
end


function compute_value(lgfa::LinearGlobalFunctionApproximator, v::AbstractArray{T}) where T
    return dot(lgfa.weights, v)
end