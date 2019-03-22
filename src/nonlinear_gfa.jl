mutable struct NonlinearGlobalFunctionApproximator{M,O,L} <: GlobalFunctionApproximator
    model::M
    optimizer::O
    loss::L
end

function fit!(ngfa::NonlinearGlobalFunctionApproximator, dataset_input::M,
              dataset_output::V) where { M <: AbstractMatrix{Float64}, V <: AbstractVector{Float64} }
    # TODO: Create `dataset` combining input and output?
    Flux.train!(params(ngfa.model), ngfa.loss, dataset, ngfa.optimizer)
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