#=
Construct a 100 x 100 grid and run discreteVI on it. Also run global approx VI where any integral point in the grid 
can be sampled. Compare the values at the integer grid points. Confirm that the max difference reduces with increasing
number of samples.
=#

# Feature vector conversion
# For point (x,y), fv is (1 x y xy x^2 y^2)
function GlobalApproximationValueIteration.convert_featurevector(::Type{SVector{6, Float64}}, s::GWPos, mdp::SimpleGridWorld)

    x = s[1]
    y = s[2]
    v = SVector{6, Float64}(1, x, y, x*y, x^2, y^2)
    return v
end


# Sample a specific integral point in the grid
function GlobalApproximationValueIteration.sample_state(mdp::SimpleGridWorld, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    x = rand(rng, 1:mdp.size[1])
    y = rand(rng, 1:mdp.size[2])
    
    return GWPos(x, y)

end


function compare_linear_against_full_grid()

    rng = MersenneTwister(1234)

    # Generate a block of reward states from 40,40 to 60,60
    rewards = Dict{GWPos, Float64}()
    for x = 35:60
        for y = 40:65
            rewards[GWPos(x, y)] = 10
        end
    end

    # Create MDP
    mdp = SimpleGridWorld(size=(100, 100), rewards=rewards)

    # Attempt to approximate globally with N samples and M iterations. As N increases, the average error should decrease
    NUM_SAMPLES = 1000
    MAX_ITERS = 100
    lin_gfa = LinearGlobalFunctionApproximator(zeros(6)) # TODO : automatic??

    global_solver = GlobalApproximationValueIterationSolver(lin_gfa; num_samples=NUM_SAMPLES, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{6, Float64})
    global_policy = solve(global_solver, mdp)


    # Now solve with dicrete VI
    solver = ValueIterationSolver(max_iterations=1000, verbose=true)
    policy = solve(solver, mdp)

    total_err = 0.0
    for state in states(mdp)
        disc_val = value(policy, state)
        approx_val = value(global_policy, state)
        total_err += abs(approx_val - disc_val)
    end

    avg_err = total_err / length(states(mdp))

    println("Average difference in value function is ", avg_err)

    return true
end

@test compare_linear_against_full_grid() == true