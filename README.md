[![Build status](https://github.com/JuliaPOMDP/GlobalApproximationValueIteration.jl/workflows/CI/badge.svg)](https://github.com/JuliaPOMDP/GlobalApproximationValueIteration.jl/actions)
[![Coverage Status](https://codecov.io/gh/JuliaPOMDP/GlobalApproximationValueIteration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/GlobalApproximationValueIteration.jl?branch=master)


# GlobalApproximationValueIteration.jl

This package implements the Global Approximation Value Iteration algorithm in Julia for solving
Markov Decision Processes (MDPs) with global function approximation.
It is functionally very similar to the previously released
[LocalApproximationValueIteration.jl](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl)
and interested users can refer to its README for more details.
The user should define the POMDP problem according to the API in
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Examples of problem definitions can be found in
[POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl).

## Installation

You need to have [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) already and the JuliaPOMDP registry added (see the README of POMDPs.jl).
Thereafter, you can add GlobalApproximationValueIteration from the package manager
```julia
using Pkg
Pkg.add("GlobalApproximationValueIteration")
```

## How it Works

This solver is one example of _Approximate Dynamic Programming_, which tries to find approximately optimal
value functions and policies for large or continuous state spaces.
As the name suggests, global approximation value iteration tries to approximate the value function over the
entire state space using a compact representation. The quality of the approximation varies with the
kind of function approximation scheme used; this repository can accommodate both linear (with feature vectors) and nonlinear
schemes. Please see **Section 4.5.1** of the book [Decision Making Under Uncertainty : Theory and Application](https://dl.acm.org/citation.cfm?id=2815660)
and **Chapter 3** of [Markov Decision Processes in Artificial Intelligence](https://books.google.co.in/books?hl=en&lr=&id=2J8_-O4-ABIC&oi=fnd&pg=PT8&dq=markov+decision+processes+in+AI&ots=mcxpyqiv0X&sig=w-gF6nzm3JxgutcslIbUDD0dAXY) for more.

## State Space Representation

The Global Approximation solver needs two things in particular from the state space of the MDP. First, it should be able to sample a state
from the state space (whether discrete or continuous). During value iteration, in each step, the solver will sample several states, estimate the value
at them and try to fit the approximation scheme. 
Second, a state instance should be representable as a _feature vector_ which will be used for linear or non-linear function approximation.
In the default case, the `feature` can just be the vector encoding of the state (see *State Space Representation* in the [README](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl)
of LocalApproximationValueIteration.jl for more on this).

## Usage 

Please refer to the [README](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl)
of LocalApproximationValueIteration.jl as the usage of the global variant is very similar to that one.
A simple example is also provided in the `test/` folder for each of linear and nonliner function approximation.
`POMDPs.jl` has a macro `@requirements_info` that determines the functions necessary to use some solver on some specific MDP model.
Other than the typical methods required for approximate value iteration and state space representation mentioned above,
the solver also requires a `GlobalFunctionApproximator` object (see `src/global_function_approximation.jl` for details
on the interface). We have also implemented two examplar approximations, linear and non-linear.
The following code snippet from `test/test_with_linear_gfa.jl` is the most relevant chunk of code
for using the solver correctly.

```julia
# Create the MDP for a typical grid world
mdp = SimpleGridWorld(size=(SIZE_X, SIZE_Y), rewards=rewards)

# Create the linear function approximation with 10 weight parameters, initialized to zero
lin_gfa = LinearGlobalFunctionApproximator(zeros(10))

# Initialize the global approximation solver with the linear approximator and solve the MDP to obtain the policy
gfa_solver = GlobalApproximationValueIterationSolver(lin_gfa, num_samples=NUM_SAMPLES, max_iterations=MAX_ITERS, verbose=true, fv_type=SVector{10, Float64})
gfa_policy = solve(gfa_solver, mdp)
```
