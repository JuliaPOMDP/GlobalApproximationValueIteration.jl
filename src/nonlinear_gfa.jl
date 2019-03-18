mutable struct NonlinearGlobalFunctionApproximator <: GlobalFunctionApproximator
    model::??
    optimizer::??
end

function fit!(ngfa::NonlinearGlobalFunctionApproximator, dataset_input::AbstractMatrix{Float64},
              dataset_output::AbstractVector{Float64})
    # TODO: Create `dataset` combining input and output?
    Flux.train!(params(ngfa.model), mse, dataset, ngfa.optimizer)
end

function compute_value(ngfa::NonlinearGlobalFunctionApproximator, v::AbstractVector{Float64})
    # TODO: What's the generic way for this?

end

function compute_value(ngfa::NonlinearGlobalFunctionApproximator, 
                        v_list::AbstractVector{V}) where V <: AbstractVector{Float64}
    @assert length(v_list) > 0
    vals = [compute_value(ngfa,v) for v in v_list]
    return vals
end