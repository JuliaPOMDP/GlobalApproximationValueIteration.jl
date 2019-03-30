mutable struct NonlinearGlobalFunctionApproximator{M,O,L} <: GlobalFunctionApproximator
    model::M
    optimizer::O
    loss::L
end

function fit!(ngfa::NonlinearGlobalFunctionApproximator, dataset_input::M,
              dataset_output::V) where { M <: AbstractMatrix{Float64}, V <: AbstractVector{Float64} }
    # Create loss function with loss type
    loss(x, y) = ngfa.loss(ngfa.model(x), y)

    # TODO: Can you confirm this is necessary? Flux seems to require data as a vector of tuples
    # https://fluxml.ai/Flux.jl/stable/training/training/#Datasets-1
    num_samples = length(dataset_output)
    data = [(dataset_input[i,:], dataset_output[i]) for i = 1:num_samples]

    Flux.train!(ngfa.loss, params(ngfa.model), data, ngfa.optimizer)
end

function compute_value(ngfa::NonlinearGlobalFunctionApproximator, v::AbstractVector{Float64})
    # TODO: Is this the correct API?
    return ngfa.model(v)
end

function compute_value(ngfa::NonlinearGlobalFunctionApproximator, 
                        v_list::AbstractVector{V}) where V <: AbstractVector{Float64}
    @assert length(v_list) > 0
    vals = [compute_value(ngfa,v) for v in v_list]
    return vals
end