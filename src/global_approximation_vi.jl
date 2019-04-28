mutable struct GlobalApproximationValueIterationSolver{GFA <: GlobalFunctionApproximator, RNG <: AbstractRNG, F} <: Solver
    gfa::GFA
    num_samples::Int64
    belres::Float64
    num_iterations::Int64
    verbose::Bool
    rng::RNG
    is_mdp_generative::Bool
    n_generative_samples::Int64
    fv_type::F
end

function GlobalApproximationValueIterationSolver(gfa::GFA; num_samples::Int64, belres::Float64=1e-3,
                                                 max_iterations::Int64=1000, verbose::Bool=false,
                                                 rng::RNG=Random.GLOBAL_RNG, is_mdp_generative::Bool=false,
                                                 n_generative_samples::Int64=0, fv_type::F=Vector{Float64}) where {GFA <: GlobalFunctionApproximator, F, RNG <: AbstractRNG}
    return GlobalApproximationValueIterationSolver(gfa, num_samples, belres, max_iterations, verbose, rng, is_mdp_generative, n_generative_samples, fv_type)
end

function GlobalApproximationValueIterationSolver()
    throw(ArgumentError("GlobalApproximationValueIterationSolver needs a GlobalFunctionApproximator object for construction!"))
end

mutable struct GlobalApproximationValueIterationPolicy{GFA <: GlobalFunctionApproximator, RNG <: AbstractRNG, F} <: Policy
    gfa::GFA
    action_map::Vector
    mdp::Union{MDP,POMDP}
    is_mdp_generative::Bool
    n_generative_samples::Int64
    fv_type::F
    rng::RNG
end

function GlobalApproximationValueIterationPolicy(mdp::Union{MDP,POMDP},
                                                 solver::GlobalApproximationValueIterationSolver)
    return GlobalApproximationValueIterationPolicy(deepcopy(solver.gfa), ordered_actions(mdp), mdp,
                                                   solver.is_mdp_generative, solver.n_generative_samples, solver.fv_type, solver.rng)
end

function convert_featurevector(::Type{V}, s::S, mdp::Union{MDP,POMDP}) where {V <: AbstractArray, S}
    return convert_s(V, s, mdp)
end

# If global function approximator is non-linear, a default convert_s is required
@POMDP_require convert_featurevector(t::Type{V} where {V <: AbstractArray}, s::S where S, mdp::Union{MDP,POMDP}) begin
    @req convert_s(::Type{V} where {V <: AbstractArray}, ::S, ::typeof(mdp))
end

@POMDP_require solve(solver::GlobalApproximationValueIterationSolver, mdp::Union{MDP,POMDP}) begin

    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req n_actions(::P)
    @subreq ordered_actions(mdp)

    @req actionindex(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    a = first(as)

    # Need to be able to sample states
    @req sample_state(::P, ::typeof(solver.rng))

    # Have different requirements depending on whether solver MDP is generative or explicit
    if solver.is_mdp_generative
        @req generate_sr(::P, ::S, ::A, ::typeof(solver.rng))
    else
        @req transition(::P, ::S, ::A)
        ss = sample_state(mdp, solver.rng)
        dist = transition(mdp, ss, a)
        D = typeof(dist)
        @req support(::D)
    end

    # Feature vector conversion must be defined either directly or by default (through convert_s)
    @subreq convert_featurevector(solver.fv_type, sample_state(mdp, solver.rng), mdp)
end

function POMDPs.solve(solver::GlobalApproximationValueIterationSolver, mdp::Union{MDP,POMDP})

    @warn_requirements solve(solver,mdp)

    # Ensure that generative model has a non-zero number of samples
    if solver.is_mdp_generative
        @assert solver.n_generative_samples > 0
    end

    # Solver parameters
    belres = solver.belres
    num_iterations = solver.num_iterations
    num_samples = solver.num_samples
    discount_factor = discount(mdp)
    gfa_type = typeof(solver.gfa)

    # Initialize the policy
    policy = GlobalApproximationValueIterationPolicy(mdp, solver)

    total_time = 0.0
    iter_time = 0.0

    temp_s = sample_state(mdp, solver.rng)
    state_dim = length(convert_featurevector(solver.fv_type, temp_s, mdp))


    for iter = 1:num_iterations

        # TODO : Is this correct
        residual = 0.0

        # Setup input and outputs for fit functions
        state_matrix = zeros(eltype(solver.fv_type), num_samples, state_dim)
        val_vector = zeros(eltype(solver.fv_type), num_samples)

        iter_time = @elapsed begin

        for i = 1:num_samples

            s = sample_state(mdp, solver.rng)
            
            pt = convert_featurevector(solver.fv_type, s, mdp)
            state_matrix[i,:] = pt

            sub_aspace = actions(mdp,s)

            if isterminal(mdp, s)
                val_vector[i] = 0.0
            else
                old_util = value(policy, s)
                max_util = -Inf

                for a in sub_aspace
                    iaction = actionindex(mdp,a)
                    u = 0.0

                    if solver.is_mdp_generative
                        for j in 1:solver.n_generative_samples
                            sp, r = generate_sr(mdp, s, a, solver.rng)
                            u += r

                            if !isterminal(mdp,sp)
                                sp_feature = convert_featurevector(solver.fv_type, sp, mdp)
                                u += p * (discount_factor*compute_value(policy.gfa, sp_feature))
                            end
                        end
                        u = u / solver.n_generative_samples
                    else
                        dist = transition(mdp,s,a)
                        for (sp, p) in weighted_iterator(dist)
                            p == 0.0 ? continue : nothing
                            r = reward(mdp, s, a, sp)
                            u += p*r

                            # Only interpolate sp if it is non-terminal
                            if !isterminal(mdp,sp)
                                sp_feature = convert_featurevector(solver.fv_type, sp, mdp)
                                u += p * (discount_factor*compute_value(policy.gfa, sp_feature))
                            end
                        end
                    end

                    max_util = (u > max_util) ? u : max_util
                    util_diff = abs(max_util - old_util)
                    util_diff > residual ? (residual = util_diff) : nothing
                end #action

                val_vector[i] = max_util
            end
        end
        # Now fit!
        fit!(policy.gfa, state_matrix, val_vector)

        end # time

        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", iter, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing
    end

    return policy
end

# TODO: LOT OF OVERLAP between below fns and those for LocalApproxVI - any way to make compact?
function POMDPs.value(policy::GlobalApproximationValueIterationPolicy, s::S) where S

    s_point = convert_featurevector(policy.fv_type, s, policy.mdp)
    val = compute_value(policy.gfa, s_point)
    return val
end


# Not explicitly stored in policy - extract from value function interpolation
function POMDPs.action(policy::GlobalApproximationValueIterationPolicy, s::S) where S
    
    mdp = policy.mdp
    best_a_idx = -1
    max_util = -Inf
    sub_aspace = actions(mdp,s)
    discount_factor = discount(mdp)


    for a in sub_aspace
        
        iaction = actionindex(mdp, a)
        u = action_value(policy,s,a)

        if u > max_util
            max_util = u
            best_a_idx = iaction
        end
    end

    return policy.action_map[best_a_idx]
end


function action_value(policy::GlobalApproximationValueIterationPolicy, s::S, a::A) where {S,A}

    mdp = policy.mdp
    discount_factor = discount(mdp)
    u = 0.0

    # As in solve(), do different things based on whether 
    # mdp is generative or explicit
    if policy.is_mdp_generative
        for j in 1:policy.n_generative_samples
            sp, r = generate_sr(mdp, s, a, policy.rng)
            sp_point = convert_featurevector(policy.fv_type, sp, mdp)
            u += r + discount_factor*compute_value(policy.gfa, sp_point)
        end
        u = u / policy.n_generative_samples
    else
        dist = transition(mdp,s,a)
        for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            r = reward(mdp, s, a, sp)
            u += p*r

            # Only interpolate sp if it is non-terminal
            if !isterminal(mdp,sp)
                sp_point = convert_featurevector(policy.fv_type, sp, mdp)
                u += p*(discount_factor*compute_value(policy.gfa, sp_point))
            end
        end
    end

    return u
end