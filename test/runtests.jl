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

Random.seed!(1234)

@testset "all" begin

    @testset "integration" begin
        include("test_with_linear_gfa.jl")
        include("test_with_nonlinear_gfa.jl")
    end
end