using POMDPModels
using POMDPs
using POMDPModelTools
using StaticArrays
using Random
using DiscreteValueIteration
using GlobalApproximationValueIteration
using Flux
using Test

@testset "all" begin

    @testset "integration" begin
        include("test_with_linear_gfa.jl")
    end
end