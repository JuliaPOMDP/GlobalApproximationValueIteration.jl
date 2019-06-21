mutable struct NonlinearGlobalFunctionApproximator{M,O,L} <: GlobalFunctionApproximator
    model::M
    optimizer::O
    loss::L
end

function fit!(ngfa::NonlinearGlobalFunctionApproximator, dataset_input::AbstractMatrix{T},
              dataset_output::AbstractArray{T}) where T
    # Create loss function with loss type
    loss(x, y) = ngfa.loss(ngfa.model(x), y)

    # NOTE : Minibatch update; 1 update to model weights
    # data = repeated((param(transpose(dataset_input)), param(transpose(dataset_output))), 1)
    data = repeated((transpose(dataset_input), transpose(dataset_output)), 1)

    Flux.train!(loss, params(ngfa.model), data, ngfa.optimizer)
end

function compute_value(ngfa::NonlinearGlobalFunctionApproximator, state_vector::AbstractArray{T}) where T
    return Flux.data(ngfa.model(state_vector))[1]
end