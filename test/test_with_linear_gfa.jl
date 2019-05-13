#=
Construct a 100 x 100 grid and run discreteVI on it. Also run global approx VI where any integral point in the grid 
can be sampled. Compare the values at the integer grid points. Confirm that the max difference reduces with increasing
number of samples.
=#

# Feature vector conversion
# For point (x,y), fv is (1 x y xy x^2 y^2)
function GlobalApproximationValueIteration.convert_featurevector(::Type{SVector{10, Float64}}, s::GWPos, mdp::SimpleGridWorld)

    x = s[1]
    y = s[2]

    v = SVector{10, Float64}(1, x, y, x*y, x^2, y^2, x^3, x^2*y, x*y^2, y^3)

    return v
end


# Sample a specific integral point in the grid
function GlobalApproximationValueIteration.sample_state(mdp::SimpleGridWorld, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    x = rand(rng, 1:mdp.size[1])
    y = rand(rng, 1:mdp.size[2])
    
    return GWPos(x, y)
end


function test_against_full_grid()

    rng = MersenneTwister(5678)

    # Generate a block of reward states from 40,40 to 60,60
    rewards = Dict{GWPos, Float64}()
    for x = 15:30
        for y = 20:35
            rewards[GWPos(x, y)] = 10
        end
    end

    # Create MDP
    mdp = SimpleGridWorld(size=(50, 50), rewards=rewards)

    # Attempt to approximate globally with N samples and M iterations. As N increases, the average error should decrease
    MAX_ITERS = 100

    NUM_SAMPLES_LOW = 100
    NUM_SAMPLES_HI = 1000

    lin_gfa_1 = LinearGlobalFunctionApproximator(zeros(10)) # TODO : automatic??
    lin_gfa_2 = LinearGlobalFunctionApproximator(zeros(10))

    solver_low = GlobalApproximationValueIterationSolver(lin_gfa_1; num_samples=NUM_SAMPLES_LOW, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{10, Float64})
    solver_hi = GlobalApproximationValueIterationSolver(lin_gfa_2; num_samples=NUM_SAMPLES_HI, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{10, Float64})

    policy_low = solve(solver_low, mdp)
    policy_hi = solve(solver_hi, mdp)

    # Now solve with dicrete VI
    solver = ValueIterationSolver(max_iterations=1000, verbose=true)
    policy = solve(solver, mdp)

    total_err_low = 0.0
    total_err_hi = 0.0

    for state in states(mdp)

        full_val = value(policy, state)
        
        approx_val_low = value(policy_low, state)
        approx_val_hi = value(policy_hi, state)   

        total_err_low += abs(full_val-approx_val_low)
        total_err_hi += abs(full_val-approx_val_hi)
    end
    
    avg_err_low = total_err_low / length(states(mdp))
    avg_err_hi = total_err_hi / length(states(mdp))

    @show avg_err_low
    @show avg_err_hi

    return (avg_err_low > avg_err_hi)
end

@test test_against_full_grid() == true