using POMDPModels
using POMDPs
using POMDPTools
using StaticArrays
using Statistics
using Random
using DiscreteValueIteration
using GlobalApproximationValueIteration
using Test

Random.seed!(1234)

@testset "all" begin

    @testset "integration" begin
        include("test_with_linear_gfa.jl")
        include("test_with_nonlinear_gfa.jl")
    end
end
