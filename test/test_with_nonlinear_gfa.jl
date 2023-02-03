
# Sample a specific integral point in the grid
function GlobalApproximationValueIteration.sample_state(mdp::SimpleGridWorld, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    x = rand(rng, 1:mdp.size[1])
    y = rand(rng, 1:mdp.size[2])
    
    return GWPos(x, y)
end


function test_against_full_grid()

    rng = MersenneTwister(2378)

    # Attempt to approximate globally with N samples and M iterations. As N increases, the average error should decrease
    MAX_ITERS = 100
    NUM_SAMPLES_LOW = 50
    NUM_SAMPLES_HI = 500

    # Grid probabilities
    SIZE_X = 10
    SIZE_Y = 10
    REWARD_COV_PROB = 0.3

    # Generate a block of reward states from 40,40 to 60,60
    rewards = Dict{GWPos, Float64}()
    for x = 1:SIZE_X
        for y = 1:SIZE_Y
            if rand(rng) < REWARD_COV_PROB
                rewards[GWPos(x, y)] = 10
            end
        end
    end

    # Create MDP
    mdp = SimpleGridWorld(size=(SIZE_X, SIZE_Y), rewards=rewards)
    

    # Define learning model
    model1 = Chain(
                Dense(2, 10, relu),
                Dense(10, 5, relu),
                Dense(5, 1))
    model2 = Chain(
                Dense(2, 10, relu),
                Dense(10, 5, relu),
                Dense(5, 1))
    opt = Adam(0.001)

    nonlin_gfa_1 = NonlinearGlobalFunctionApproximator(model1, opt, Flux.mse)
    nonlin_gfa_2 = NonlinearGlobalFunctionApproximator(model2, opt, Flux.mse)

    solver_low = GlobalApproximationValueIterationSolver(nonlin_gfa_1; num_samples=NUM_SAMPLES_LOW, max_iterations=MAX_ITERS, verbose=true)
    solver_hi = GlobalApproximationValueIterationSolver(nonlin_gfa_2; num_samples=NUM_SAMPLES_HI, max_iterations=MAX_ITERS, verbose=true)

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

@test_broken test_against_full_grid() == true
