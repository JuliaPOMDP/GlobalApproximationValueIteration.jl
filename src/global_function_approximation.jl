abstract type GlobalFunctionApproximator end


# TODO: Specify that the below does not assume a loss, but we implement the defaults for LSE loss?
"""
    fit!(gfa::GlobalFunctionApproximator, dataset_input::AbstractMatrix, dataset_output::AbstractVector)
Fit the global function approximator to the dataset using some optimization method and a chosen
loss function.
"""
function fit! end

"""
    compute_value(gfa::GlobalFunctionApproximator, v::AbstractVector)
Return the value of the function at some query point v, based on the global function approximator

    compute_value(gfa::GlobalFunctionApproximator, v_list::AbstractVector{V}) where V <: AbstractVector{Float64}
Return the value of the function for a list of query points, based on the global function approximator
"""
function compute_value end