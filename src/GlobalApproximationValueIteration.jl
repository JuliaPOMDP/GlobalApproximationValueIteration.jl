module GlobalApproximationValueIteration

# Stdlib imports
using LinearAlgebra
using Random
using Printf

# POMDPs imports
using POMDPs
using POMDPPolicies
using POMDPModels
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
    compute_featurevector,
    solve,
    action,
    value,
    action_value


include("global_function_approximation.jl")
include("linear_gfa.jl")
include("nonlinear_gfa.jl")
include("global_approximation_vi.jl")

end # module