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


function test_absolute_error()

    rng = MersenneTwister(1234)

    MAX_ITERS = 500
    NUM_SAMPLES = 1000

    SIZE_X = 5
    SIZE_Y = 5
    REWARD_COV_PROB = 0.4

    rewards = Dict{GWPos, Float64}()
    for x = 1:SIZE_X
        for y = 1:SIZE_Y
            if rand(rng) < REWARD_COV_PROB
                rewards[GWPos(x, y)] = 1
            end
        end
    end

    # Create MDP
    mdp = SimpleGridWorld(size=(SIZE_X, SIZE_Y), rewards=rewards)

    lin_gfa = LinearGlobalFunctionApproximator(zeros(10))
    gfa_solver = GlobalApproximationValueIterationSolver(lin_gfa, num_samples=NUM_SAMPLES, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{10, Float64})
    gfa_policy = solve(gfa_solver, mdp)

    solver = ValueIterationSolver(max_iterations=1000, verbose=true)
    policy = solve(solver, mdp)

    error_arr = Vector{Float64}(undef, 0)

    for state in states(mdp)

        full_val = value(policy, state)
        approx_val = value(gfa_policy, state)
        abs_diff = abs(full_val - approx_val)

        push!(error_arr, abs_diff)
    end

    @show mean(error_arr)
    @show maximum(error_arr)

    return (mean(error_arr) < 0.04 && maximum(error_arr) < 0.35)
end


function test_relative_error()

    rng = MersenneTwister(2378)

    # Attempt to approximate globally with N samples and M iterations. As N increases, the average error should decrease
    MAX_ITERS = 500
    NUM_SAMPLES_LOW = 30
    NUM_SAMPLES_HI = 1000

    # Grid probabilities
    SIZE_X = 5
    SIZE_Y = 5
    REWARD_COV_PROB = 0.4

    # Generate a block of reward states from 40,40 to 60,60
    rewards = Dict{GWPos, Float64}()
    for x = 1:SIZE_X
        for y = 1:SIZE_Y
            if rand(rng) < REWARD_COV_PROB
                rewards[GWPos(x, y)] = 1
            end
        end
    end

    # Create MDP
    mdp = SimpleGridWorld(size=(SIZE_X, SIZE_Y), rewards=rewards)
    

    lin_gfa_1 = LinearGlobalFunctionApproximator(zeros(10))
    lin_gfa_2 = LinearGlobalFunctionApproximator(zeros(10))

    solver_low = GlobalApproximationValueIterationSolver(lin_gfa_1; num_samples=NUM_SAMPLES_LOW, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{10, Float64})
    solver_hi = GlobalApproximationValueIterationSolver(lin_gfa_2; num_samples=NUM_SAMPLES_HI, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{10, Float64})

    policy_low = solve(solver_low, mdp)
    policy_hi = solve(solver_hi, mdp)

    # Now solve with dicrete VI
    solver = ValueIterationSolver(max_iterations=1000, verbose=true)
    policy = solve(solver, mdp)

    err_arr_low = Vector{Float64}(undef, 0)
    err_arr_hi = Vector{Float64}(undef, 0)

    for state in states(mdp)

        full_val = value(policy, state)
        
        approx_val_low = value(policy_low, state)
        approx_val_hi = value(policy_hi, state)   

        push!(err_arr_low, abs(full_val - approx_val_low))
        push!(err_arr_hi, abs(full_val - approx_val_hi))

    end

    @show mean(err_arr_low), mean(err_arr_hi)
    @show maximum(err_arr_low), maximum(err_arr_hi)

    return (mean(err_arr_low) > mean(err_arr_hi) && maximum(err_arr_low) > maximum(err_arr_hi))
end


# @test test_absolute_error() == true
@test test_relative_error() == true