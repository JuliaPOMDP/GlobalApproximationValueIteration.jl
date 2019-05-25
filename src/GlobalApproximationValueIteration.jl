module GlobalApproximationValueIteration

using Base.Iterators: repeated


# Stdlib imports
using LinearAlgebra
using Random
using Printf

# For function approximation
using MultivariateStats
using Flux

# POMDPs imports
using POMDPs
using POMDPPolicies
using POMDPModelTools

export
    GlobalFunctionApproximator,
    fit!,
    compute_value,
    LinearGlobalFunctionApproximator,
    NonlinearGlobalFunctionApproximator

export
    GlobalApproximationValueIterationSolver,
    GlobalApproximationValueIterationPolicy,
    convert_featurevector,
    sample_state

function sample_state end

include("global_function_approximation.jl")
include("linear_gfa.jl")
include("nonlinear_gfa.jl")
include("global_approximation_vi.jl")

end # module