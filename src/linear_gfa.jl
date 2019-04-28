mutable struct LinearGlobalFunctionApproximator{W <: AbstractArray} <: GlobalFunctionApproximator
    weights::W
end

function fit!(lgfa::LinearGlobalFunctionApproximator, dataset_input::AbstractMatrix{T},
              dataset_output::AbstractArray{T}) where T
    # TODO: Since we are ASSIGNING to weights here, does templating even matter? Does the struct even matter?
    lgfa.weights = llsq(dataset_input, dataset_output, bias=false)
end


function compute_value(lgfa::LinearGlobalFunctionApproximator, v::AbstractArray{T}) where T
    return dot(lgfa.weights, v)
end