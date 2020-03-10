using Revise
using POMDPModels
using POMDPs
using POMDPModelTools
using StaticArrays
using Random
using DiscreteValueIteration
using GlobalApproximationValueIteration
using Flux
using Statistics
using Test

include("test/test_with_nonlinear_gfa.jl")
include("test/test_with_linear_gfa.jl")