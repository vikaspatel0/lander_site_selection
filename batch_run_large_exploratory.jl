using POMDPs
using POMDPTools
using DiscreteValueIteration
using Distributions
using Random
using MAT
using Statistics
using DelimitedFiles
using Distributions: pdf, cdf, quantile, Normal

# =========================
# Risk / percentile modes
# =========================

@enum RiskMode begin
    RiskConstP
    RiskEntropicSigma
    RiskEVaR
    RiskCVaR
    Thompson
end

@enum SigmaRefMode begin
    SigmaMax
    SigmaMean
    SigmaMin
    Cellwise
end

# =========================
# Policy / planning modes
# =========================

@enum PlannerMode begin
    PlannerVI           # solve MDP via value iteration each step (original)
    PlannerGreedyTarget # choose target cell by maximizing landing score, then move 1 step toward it
end

# =========================
# Utilities
# =========================

manhattan(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = abs(a[1]-b[1]) + abs(a[2]-b[2])

function entropic_percentile(beta::Float64, sigma_ref::Float64)
    cdf(Normal(0,1), -(beta/2.0) * sigma_ref)
end

function risk_sensitive_reward(mean::Float64, std::Float64, percentile::Float64)
    if std ≈ 0.0
        return mean
    end
    quantile(Normal(mean, std), percentile)
end

function greedy_step_toward(pos::Tuple{Int,Int}, target::Tuple{Int,Int})
    i,j = pos
    ti,tj = target
    di = ti - i
    dj = tj - j
    if abs(di) >= abs(dj) && di != 0
        return di > 0 ? :down : :up
    elseif dj != 0
        return dj > 0 ? :right : :left
    else
        return :none
    end
end

function reachable_indices_fast(nrows::Int, ncols::Int, i::Int, j::Int, steps::Int)
    inds = Tuple{Int,Int}[]
    sizehint!(inds, (steps + 1)^2 + steps^2) # Pre-allocate memory for performance

    # Iterate through each possible row in the diamond's span
    for ii in max(1, i - steps):min(nrows, i + steps)
        # For the current row `ii`, calculate how many steps we can take horizontally.
        # This is the total steps minus the steps we already used vertically.
        di = abs(ii - i)
        horizontal_reach = steps - di

        # Determine the start and end column indices for this row's slice of the diamond
        # and clamp them to the grid boundaries.
        j_start = max(1, j - horizontal_reach)
        j_end = min(ncols, j + horizontal_reach)

        # Now, we only have to add the cells in this valid horizontal slice
        for jj in j_start:j_end
            push!(inds, (ii, jj))
        end
    end
    return inds
end

function select_sigma_ref(grid_std::Matrix{Float64},
                          reachable::Vector{Tuple{Int,Int}},
                          mode::SigmaRefMode)
    sigmas = [grid_std[ii, jj] for (ii, jj) in reachable]

    if mode == SigmaMax
        return maximum(sigmas)
    elseif mode == SigmaMean
        return mean(sigmas)
    elseif mode == SigmaMin
        return minimum(sigmas)
    else
        error("Unknown SigmaRefMode")
    end
end

function evar_normal(mean::Float64, std::Float64, alpha::Float64)
    if std ≈ 0.0
        return mean
    end

    # EVaR bound for Gaussian
    k = sqrt(2 * log(1/alpha))

    return mean - std * k
end

function cvar_normal(mean::Float64, std::Float64, alpha::Float64)
    if std ≈ 0.0
        return mean
    end
    # We use the standard normal distribution N(0,1) for the core calculation
    q = quantile(Normal(0, 1), alpha)  # This is the z-score for the alpha-percentile
    phi = pdf(Normal(0, 1), q)         # This is the height of the PDF at that z-score
    
    # The formula for CVaR of a normal distribution
    return mean - std * (phi / alpha)
end

# =========================
# Risk configuration (independent)
# =========================

Base.@kwdef struct RiskConfig
    mode::RiskMode = RiskConstP
    p_const::Float64 = 0.90           # used for RiskConstP
    beta::Float64 = 0.0               # used for RiskEntropicSigma
    sigma_ref_mode::SigmaRefMode = SigmaMean
    alpha::Float64 = 0.1              # used for RiskCVaR (e.g., worst 10% of outcomes)
end

function compute_risk_sensitive_value(
    rc::RiskConfig,
    cell_mean::Float64,
    cell_std::Float64,
    grid_std::Union{Nothing, Matrix{Float64}} = nothing,
    pos::Union{Nothing, Tuple{Int,Int}} = nothing,
    z::Union{Nothing, Int} = nothing,
    reachable::Union{Nothing, Vector{Tuple{Int,Int}}} = nothing,  # Add this
)
    if rc.mode == RiskCVaR
        return cvar_normal(cell_mean, cell_std, rc.alpha)
    elseif rc.mode == RiskEVaR
        return evar_normal(cell_mean, cell_std, rc.alpha)
    elseif rc.mode == Thompson
        return rand(Normal(cell_mean, cell_std))
    else
        p = 0.0
        if rc.mode == RiskConstP
            p = rc.p_const
        elseif rc.mode == RiskEntropicSigma
            # Use the precomputed reachable indices
            if rc.sigma_ref_mode == Cellwise
                p = entropic_percentile(rc.beta, cell_std) # <-- Simpler version using only the cell's std
            else
                sigma_ref = select_sigma_ref(grid_std, reachable, rc.sigma_ref_mode)
                p = entropic_percentile(rc.beta, sigma_ref)
            end
        else
            error("Unknown RiskMode")
        end
        return risk_sensitive_reward(cell_mean, cell_std, p)
    end
end


# =========================
# Target selection (independent)
# =========================
Base.@kwdef struct GreedyTargetConfig
    lambda_travel::Float64 = 0.01 # tradeoff vs distance
    lambda_explore::Float64 = 0.01 # tradeoff for exploration bonus
    restrict_reachable::Bool = true
end
Base.@kwdef struct PlannerConfig
    mode::PlannerMode = PlannerVI
    greedy::GreedyTargetConfig = GreedyTargetConfig()
end
Base.@kwdef struct BeliefSnapshot
    mean_grid::Matrix{Float64}
    std_grid::Matrix{Float64}
    pos::Tuple{Int,Int}
    z::Int
end
function expected_sigma_mean_gain_for_action(
    snap::BeliefSnapshot,
    sensed_state::Tuple{Int,Int,Int},
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64
)
    i_curr, j_curr, altitude = sensed_state
    nrows, ncols = size(snap.std_grid)
    w = update_weight(altitude, z_update, transition_k)
    cone_radius = altitude * tan(cone_angle)
    new_std = max(noise_sigma * (1.0 - w), 0.0)
    gain = 0.0
    for i in 1:nrows, j in 1:ncols
        dist = sqrt((i - i_curr)^2 + (j - j_curr)^2)
        if dist <= cone_radius && new_std < snap.std_grid[i, j]
            gain += (snap.std_grid[i, j] - new_std) * snap.mean_grid[i, j]
        end
    end
    return gain
end
function select_target_greedy(cfg::GreedyTargetConfig, snap::BeliefSnapshot, rc::RiskConfig,
                               reachable::Union{Nothing, Vector{Tuple{Int,Int}}} = nothing;
                               noise_sigma::Float64,
                               cone_angle::Float64,
                               z_update::Int,
                               transition_k::Float64)
    nrows, ncols = size(snap.mean_grid)
    candidates = if cfg.restrict_reachable && !isnothing(reachable)
        reachable  # Use precomputed
    elseif cfg.restrict_reachable
        reachable_indices_fast(nrows, ncols, snap.pos[1], snap.pos[2], snap.z)
    else
        [(ti,tj) for ti in 1:nrows for tj in 1:ncols]
    end
    best_score = -Inf
    best_cell = snap.pos
    # Compute exploration gain for each immediate action once.
    current_state = (snap.pos[1], snap.pos[2], snap.z)
    action_gains = Dict{Symbol, Float64}()
    for a in (:up, :down, :left, :right, :none)
        sensed_state = step_next_state(nrows, ncols, current_state, a)
        action_gains[a] = expected_sigma_mean_gain_for_action(
            snap, sensed_state, noise_sigma, cone_angle, z_update, transition_k
        )
    end
    for (ti,tj) in candidates
        v = compute_risk_sensitive_value(rc, snap.mean_grid[ti,tj], snap.std_grid[ti,tj],
                                        snap.std_grid, snap.pos, snap.z, candidates)
        d = manhattan(snap.pos, (ti,tj))
        a = greedy_step_toward(snap.pos, (ti,tj))
        score = v - cfg.lambda_travel*d + cfg.lambda_explore*action_gains[a]
        if score > best_score
            best_score = score
            best_cell = (ti,tj)
        end
    end
    return best_cell
end
# =========================
# MDP (unchanged core)
# =========================

struct UncertainGridDescentMDP{F} <: MDP{Tuple{Int,Int,Int}, Symbol}
    grid_mean::Matrix{Float64}
    grid_std::Matrix{Float64}
    max_altitude::Int
    risk_cfg::RiskConfig # <-- Add this
    percentile_fn::F # This is now only for percentile-based modes
    sigma_ref::Float64
    cone_angle::Float64
end

function POMDPs.states(mdp::UncertainGridDescentMDP)
    vec([(i, j, z) for i in 1:size(mdp.grid_mean,1),
                    j in 1:size(mdp.grid_mean,2),
                    z in 0:mdp.max_altitude])
end

function POMDPs.stateindex(mdp::UncertainGridDescentMDP, s::Tuple{Int,Int,Int})
    i, j, z = s
    nrows = size(mdp.grid_mean, 1)
    ncols = size(mdp.grid_mean, 2)
    (i - 1) + (j - 1) * nrows + z * nrows * ncols + 1
end

POMDPs.actions(::UncertainGridDescentMDP) = [:up, :down, :left, :right, :none]
POMDPs.actionindex(::UncertainGridDescentMDP, a::Symbol) =
    findfirst(==(a), [:up, :down, :left, :right, :none])

POMDPs.isterminal(::UncertainGridDescentMDP, s::Tuple{Int,Int,Int}) = s[3] == 0

function POMDPs.transition(mdp::UncertainGridDescentMDP, s::Tuple{Int,Int,Int}, a::Symbol)
    i, j, z = s
    z_next = max(z - 1, 0)
    next = if a == :up
        (max(i-1,1), j, z_next)
    elseif a == :down
        (min(i+1,size(mdp.grid_mean,1)), j, z_next)
    elseif a == :left
        (i, max(j-1,1), z_next)
    elseif a == :right
        (i, min(j+1,size(mdp.grid_mean,2)), z_next)
    else
        (i, j, z_next)
    end
    Deterministic(next)
end

function POMDPs.reward(mdp::UncertainGridDescentMDP, s, a, sp)
    i, j, z_next = sp
    action_penalty = (a == :none) ? 0.0 : -0.01
    if z_next == 0
        landing_reward = compute_risk_sensitive_value(
            mdp.risk_cfg,
            mdp.grid_mean[i, j],
            mdp.grid_std[i, j],
            mdp.grid_std,
            (i, j),
            z_next
        )
        return landing_reward + action_penalty
    else
        return action_penalty
    end
end

POMDPs.discount(::UncertainGridDescentMDP) = 1.0

# =========================
# Terrain
# =========================

function generate_terrain_2(size::Int; seed=42, value_min=-10, value_range=20)
    Random.seed!(seed)
    x = range(0, 4π, length=size)
    y = range(0, 4π, length=size)
    terrain = zeros(size, size)

    # Randomize the coefficients and phase shifts for each layer
    for _ in 1:3 # Generate 3 random layers
        coeff = 0.1 + 0.3 * rand()     # Random coefficient
        freq_x = 1 + 3 * rand()        # Random frequency for x
        freq_y = 1 + 3 * rand()        # Random frequency for y
        phase_x = 2π * rand()          # Random phase shift for x
        phase_y = 2π * rand()          # Random phase shift for y
        
        for i in 1:size, j in 1:size
            terrain[i,j] += coeff * sin(freq_x * x[i] + phase_x) * cos(freq_y * y[j] + phase_y)
        end
    end
    
    # Add final, smaller random noise
    terrain .+= 0.05 * randn(size, size)

    # Normalize the terrain to the desired range
    # (This part of your code was good, no changes needed here)
    curr_min = minimum(terrain)
    curr_max = maximum(terrain)
    curr_range = curr_max - curr_min

    # Avoid division by zero if terrain is flat
    if curr_range > 0
        scale_factor = value_range / curr_range
        terrain .-= curr_min
        terrain .*= scale_factor
    end
    
    terrain .+= value_min
    
    return terrain
end

# =========================
# Cone sensing update
# =========================

function update_weight(z::Int, z_update::Int, transition_k::Float64)
    if z_update <= 0
        return 0.0
    end
    if z <= 0
        return 1.0
    elseif z >= z_update
        return 0.0
    end
    x = z / z_update
    if transition_k == 0.0
        return 1.0 - x
    elseif transition_k > 0.0
        ek  = exp(-transition_k)
        ex  = exp(-transition_k * x)
        return (ex - ek) / (1 - ek)
    else
        error("transition_k must be ≥ 0.")
    end
end

function update_with_cone!(
    grid_std::Matrix{Float64},
    mean_grid::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64
)
    i_curr, j_curr = current_pos
    nrows, ncols = size(grid_std)

    w = update_weight(altitude, z_update, transition_k)
    cone_radius = altitude * tan(cone_angle)

    new_std = max(noise_sigma * (1.0-w), 0.0)

    for i in 1:nrows, j in 1:ncols
        dist = sqrt((i - i_curr)^2 + (j - j_curr)^2)
        if dist <= cone_radius
            if new_std < grid_std[i, j]
                grid_std[i, j] = new_std
                mean_grid[i, j] = initial_mean_grid[i, j] + w * update_grid[i, j]
            end
        end
    end

    if altitude == 0
        grid_std[i_curr, j_curr] = 0.0
        mean_grid[i_curr, j_curr] = initial_mean_grid[i_curr, j_curr] + update_grid[i_curr, j_curr]
    end
    return nothing
end

# =========================
# Planner glue (keeps dynamics separate from selection)
# =========================

function step_next_state(nrows::Int, ncols::Int, s::Tuple{Int,Int,Int}, a::Symbol)
    i,j,z = s
    z_next = max(z-1, 0)
    if a == :up
        return (max(i-1,1), j, z_next)
    elseif a == :down
        return (min(i+1,nrows), j, z_next)
    elseif a == :left
        return (i, max(j-1,1), z_next)
    elseif a == :right
        return (i, min(j+1,ncols), z_next)
    else
        return (i, j, z_next)
    end
end

function plan_with_cone_sensing(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    risk_cfg::RiskConfig = RiskConfig(mode=RiskConstP, p_const=0.9),
    planner_cfg::PlannerConfig = PlannerConfig(mode=PlannerGreedyTarget),
)
    nrows, ncols = size(initial_mean_grid)
    mean_grid = copy(initial_mean_grid)
    grid_std  = fill(noise_sigma, nrows, ncols)

    prev_target = nothing

    StepT = @NamedTuple{
        state::Tuple{Int,Int,Int},
        action::Symbol,
        reward::Float64,
        next_state::Tuple{Int,Int,Int},
        grid_std_snapshot::Matrix{Float64},
        grid_mean_snapshot::Matrix{Float64},
        target::Union{Nothing,Tuple{Int,Int}}
    }
    trajectory = StepT[]
    current_state = start_state
    total_reward = 0.0

    while current_state[3] > 0
        i,j,z = current_state

        # **Compute reachable indices ONCE per timestep**
        reachable = reachable_indices_fast(nrows, ncols, i, j, z)
        
        # dynamics/sensing update (mutates belief)
        update_with_cone!(
            grid_std, mean_grid, initial_mean_grid, update_grid,
            (i,j), z, noise_sigma, cone_angle,
            z_update, transition_k
        )

        target = nothing
        action = :none
        next_state = current_state
        r = 0.0

        if planner_cfg.mode == PlannerVI
            percentile_fn = (z_, sigref_) -> 0.0
            mdp = UncertainGridDescentMDP(mean_grid, copy(grid_std), z, risk_cfg, percentile_fn, 0.0, cone_angle)

            solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=false)
            policy = solve(solver, mdp)

            action = POMDPs.action(policy, current_state)
            next_state = rand(transition(mdp, current_state, action))
            prev_target = (next_state[1], next_state[2])
            r = POMDPs.reward(mdp, current_state, action, next_state)

        elseif planner_cfg.mode == PlannerGreedyTarget
            snap = BeliefSnapshot(mean_grid=mean_grid, std_grid=grid_std, pos=(i,j), z=z)
            target = select_target_greedy(
                planner_cfg.greedy, snap, risk_cfg, reachable;
                noise_sigma=noise_sigma,
                cone_angle=cone_angle,
                z_update=z_update,
                transition_k=transition_k
            )
            action = greedy_step_toward((i,j), target)
            next_state = step_next_state(nrows, ncols, current_state, action)
            prev_target = (next_state[1], next_state[2])

            action_penalty = (action == :none) ? 0.0 : -0.01
            r = if next_state[3] == 0
                landing_reward = compute_risk_sensitive_value(
                    risk_cfg,
                    mean_grid[next_state[1], next_state[2]],
                    grid_std[next_state[1], next_state[2]],
                    grid_std,
                    (next_state[1], next_state[2]),
                    next_state[3],
                    reachable
                )
                landing_reward + action_penalty
            else
                action_penalty
            end
        else
            error("Unknown PlannerMode")
        end

        push!(trajectory, (state=current_state,
                           action=action,
                           reward=r,
                           next_state=next_state,
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=target))

        total_reward += r
        current_state = next_state
    end

    # final landing update at z=0
    update_with_cone!(
        grid_std, mean_grid, initial_mean_grid, update_grid,
        (current_state[1], current_state[2]), 0, noise_sigma, cone_angle,
        z_update, transition_k
    )

    return trajectory, total_reward, grid_std, mean_grid
end

# =========================
# Batch runner
# =========================

percentiles(v::AbstractVector{<:Real}) = (
    q25 = quantile(v, 0.25),
    q50 = quantile(v, 0.50),
    q75 = quantile(v, 0.75),
)

function run_batch(
    n_sims::Int;
    seed::Int = 1234,
    grid_size::Int = 60,
    start_state::Tuple{Int,Int,Int} = (20, 36, 30),
    cone_angle::Float64 = π/8,
    noise_sigma::Float64 = 2.0,
    terrain_value_min::Float64 = -10.0,
    terrain_value_range::Float64 = 20.0,
    z_update::Int = 20,
    transition_k::Float64 = 0.0,
    out_csv::String = "batch_results.csv",
    risk_cfg::RiskConfig = RiskConfig(mode=RiskConstP, p_const=0.9),
    planner_cfg::PlannerConfig = PlannerConfig(mode=PlannerGreedyTarget),
)
    rng = MersenneTwister(seed)

    header = ["run",
              "init_q25","init_med","init_q75",
              "final_q25","final_med","final_q75",
              "landing_value",
              "landing_i","landing_j"]

    rows = Vector{Vector{Any}}(undef, n_sims)

    for run in 1:n_sims
        terrain_seed = rand(rng, 1:10^9)
        noise_seed   = rand(rng, 1:10^9)

        initial_mean_grid = generate_terrain_2(
            grid_size;
            seed=terrain_seed,
            value_min=terrain_value_min,
            value_range=terrain_value_range
        )

        rng_noise = MersenneTwister(noise_seed)
        update_grid = noise_sigma .* randn(rng_noise, grid_size, grid_size)

        traj, total_reward, final_std, final_mean_grid = plan_with_cone_sensing(
            initial_mean_grid,
            update_grid,
            noise_sigma,
            start_state,
            cone_angle;
            z_update=z_update,
            transition_k=transition_k,
            risk_cfg=risk_cfg,
            planner_cfg=planner_cfg
        )

        init_p = percentiles(vec(initial_mean_grid))
        final_p = percentiles(vec(initial_mean_grid + update_grid))

        land_i, land_j, _ = traj[end].next_state
        landing_value = final_mean_grid[land_i, land_j]

        rows[run] = Any[
            run,
            init_p.q25, init_p.q50, init_p.q75,
            final_p.q25, final_p.q50, final_p.q75,
            landing_value,
            land_i, land_j
        ]
    end

    open(out_csv, "w") do io
        writedlm(io, reshape(header, 1, :), ',')
        for r in rows
            writedlm(io, reshape(r, 1, :), ',')
        end
    end

    return out_csv
end

# =========================
# MATLAB trajectory saver (unchanged)
# =========================

function save_trajectory_for_matlab(trajectory, grid_mean, final_std, filename="lander_trajectory.mat")
    n_steps = length(trajectory)
    nrows, ncols = size(grid_mean)

    std_history = zeros(nrows, ncols, n_steps + 1)
    mean_history = zeros(nrows, ncols, n_steps + 1)
    lander_i = zeros(Int, n_steps + 1)
    lander_j = zeros(Int, n_steps + 1)
    lander_z = zeros(Int, n_steps + 1)
    actions = fill("", n_steps)
    rewards = zeros(n_steps)

    for (idx, step) in enumerate(trajectory)
        std_history[:, :, idx] = step.grid_std_snapshot
        mean_history[:, :, idx] = step.grid_mean_snapshot
        lander_i[idx] = step.state[1]
        lander_j[idx] = step.state[2]
        lander_z[idx] = step.state[3]
        actions[idx] = string(step.action)
        rewards[idx] = step.reward
    end

    std_history[:, :, end] = final_std
    lander_i[end] = trajectory[end].next_state[1]
    lander_j[end] = trajectory[end].next_state[2]
    lander_z[end] = trajectory[end].next_state[3]

    matwrite(filename, Dict(
        "std_history" => std_history,
        "lander_i" => lander_i,
        "lander_j" => lander_j,
        "lander_z" => lander_z,
        "actions" => actions,
        "rewards" => rewards,
        "mean_history" => mean_history,
        "n_steps" => n_steps + 1,
        "grid_size" => nrows
    ))
end

transition_ks = [0.0, 2.0, 4.0, 7.0, 15.0]
transition_k_names = ["0", "2", "4", "7", "15"]
sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
sigma_names = ["0_5", "1_0", "2_0", "3_0", "5_0"]

sigma_strats_test = [SigmaMax, SigmaMean, SigmaMin, Cellwise]
sigma_strats_test_names = ["max", "mean", "min", "cellwise"]

CVar_alphas_test = [0.001]
CVar_alphas_test_names = ["0_1"]

EVaR_alphas_test = [0.9]
EVaR_alphas_test_names = ["90"]

for (k, k_name) in zip(transition_ks, transition_k_names)
    println("Starting transition strategy: k=$k_name")
    for (sigma, sigma_name) in zip(sigmas, sigma_names)
        #println("  Testing noise sigma: $sigma")
        #for (strat, name) in zip(sigma_strats_test, sigma_strats_test_names)
        #    println("Starting percentile strategy: $name")
        #    run_batch(1000; seed=1234, transition_k=k, noise_sigma=sigma, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_$(name)_perc_k$(k_name)_sigma$(sigma_name).csv", risk_cfg=RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=strat), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget, greedy=GreedyTargetConfig(lambda_explore=0.0)))
        #    run_batch(1000; seed=1234, transition_k=k, noise_sigma=sigma, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_$(name)_perc_expl_k$(k_name)_sigma$(sigma_name).csv", risk_cfg=RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=strat), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget, greedy=GreedyTargetConfig(lambda_explore=0.1)))
        #end
        println("  Testing CVaR strategy")
        for (alpha_test, alpha_name) in zip(CVar_alphas_test, CVar_alphas_test_names)
            println("Starting CVaR alpha: $alpha_test")
            run_batch(1000; seed=1234, transition_k=k, noise_sigma=sigma, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_CVaR_alpha$(alpha_name)_k$(k_name)_sigma$(sigma_name).csv", risk_cfg=RiskConfig(mode=RiskCVaR, alpha=alpha_test), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget, greedy=GreedyTargetConfig(lambda_explore=0.0)))
            run_batch(1000; seed=1234, transition_k=k, noise_sigma=sigma, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_CVaR_alpha$(alpha_name)_expl_k$(k_name)_sigma$(sigma_name).csv", risk_cfg=RiskConfig(mode=RiskCVaR, alpha=alpha_test), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget, greedy=GreedyTargetConfig(lambda_explore=0.1)))
        end
        #println("  Testing EVaR strategy")
        #for (alpha_test, alpha_name) in zip(EVaR_alphas_test, EVaR_alphas_test_names)
        #    println("Starting EVaR alpha: $alpha_test")
        #    run_batch(1000; seed=1234, transition_k=k, noise_sigma=sigma, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_EVaR_alpha$(alpha_name)_k$(k_name)_sigma$(sigma_name).csv", risk_cfg=RiskConfig(mode=RiskEVaR, alpha=alpha_test), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget, greedy=GreedyTargetConfig(lambda_explore=0.0)))
        #    run_batch(1000; seed=1234, transition_k=k, noise_sigma=sigma, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_EVaR_alpha$(alpha_name)_expl_k$(k_name)_sigma$(sigma_name).csv", risk_cfg=RiskConfig(mode=RiskEVaR, alpha=alpha_test), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget, greedy=GreedyTargetConfig(lambda_explore=0.1)))
        #end
    end
end


#println("Done with CVaR batches, starting ConstP batches...")
#
#percentiles_test = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
#percentiles_test_names = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
#
#for (p, name) in zip(percentiles_test, percentiles_test_names)
#    run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k0_sigma0_5.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k0_sigma1_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k0_sigma2_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k0_sigma3_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k0_sigma5_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#
#    run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k2_sigma0_5.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k2_sigma1_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k2_sigma2_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k2_sigma3_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k2_sigma5_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#
#    run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k4_sigma0_5.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k4_sigma1_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k4_sigma2_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k4_sigma3_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k4_sigma5_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#
#    run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k7_sigma0_5.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k7_sigma1_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k7_sigma2_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k7_sigma3_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k7_sigma5_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#
#    run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k15_sigma0_5.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k15_sigma1_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k15_sigma2_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k15_sigma3_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#    run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_const_p$(name)_test0_k15_sigma5_0.csv", risk_cfg=RiskConfig(mode=RiskConstP, p_const=p), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget)) # PlannerVI, PlannerGreedyTarget)
#
#end
#
#println("Done with ConstP batches, starting sigma-LCB")

#println("Done with sigma-LCB batches, starting Thompson batches...")
#
#run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k0_sigma0_5.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k0_sigma1_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k0_sigma2_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k0_sigma3_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=0.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k0_sigma5_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#
#run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k2_sigma0_5.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k2_sigma1_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k2_sigma2_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k2_sigma3_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=2.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k2_sigma5_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#
#run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k4_sigma0_5.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k4_sigma1_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k4_sigma2_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k4_sigma3_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=4.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k4_sigma5_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#
#run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k7_sigma0_5.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k7_sigma1_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k7_sigma2_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k7_sigma3_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=7.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k7_sigma5_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#
#run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k15_sigma0_5.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k15_sigma1_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k15_sigma2_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k15_sigma3_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))
#run_batch(1000; seed=1234, transition_k=15.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, out_csv="data_large/batch_thompson_test0_k15_sigma5_0.csv", risk_cfg=RiskConfig(mode=Thompson), planner_cfg=PlannerConfig(mode=PlannerGreedyTarget))


