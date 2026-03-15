using Distributions
using Random
using Statistics
using DelimitedFiles

# =========================
# Utilities
# =========================

const ACTIONS = [:up, :down, :left, :right, :none]

manhattan(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = abs(a[1]-b[1]) + abs(a[2]-b[2])

function step_next_state(nrows::Int, ncols::Int, s::Tuple{Int,Int,Int}, a::Symbol)
    i, j, z = s
    z_next = max(z - 1, 0)
    if a == :up
        return (max(i - 1, 1), j, z_next)
    elseif a == :down
        return (min(i + 1, nrows), j, z_next)
    elseif a == :left
        return (i, max(j - 1, 1), z_next)
    elseif a == :right
        return (i, min(j + 1, ncols), z_next)
    else
        return (i, j, z_next)
    end
end

action_penalty(a::Symbol) = (a == :none) ? 0.0 : -0.01

# =========================
# Terrain (kept equivalent)
# =========================

function generate_terrain_2(size::Int; seed=42, value_min=-10, value_range=20)
    Random.seed!(seed)
    x = range(0, 4pi, length=size)
    y = range(0, 4pi, length=size)
    terrain = zeros(size, size)

    for _ in 1:3
        coeff = 0.1 + 0.3 * rand()
        freq_x = 1 + 3 * rand()
        freq_y = 1 + 3 * rand()
        phase_x = 2pi * rand()
        phase_y = 2pi * rand()

        for i in 1:size, j in 1:size
            terrain[i, j] += coeff * sin(freq_x * x[i] + phase_x) * cos(freq_y * y[j] + phase_y)
        end
    end

    terrain .+= 0.05 * randn(size, size)

    curr_min = minimum(terrain)
    curr_max = maximum(terrain)
    curr_range = curr_max - curr_min

    if curr_range > 0
        scale_factor = value_range / curr_range
        terrain .-= curr_min
        terrain .*= scale_factor
    end

    terrain .+= value_min
    return terrain
end

# =========================
# Cone sensing update (std only for MCTS belief updates)
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
        ek = exp(-transition_k)
        ex = exp(-transition_k * x)
        return (ex - ek) / (1 - ek)
    else
        error("transition_k must be >= 0.")
    end
end

function update_with_cone_std_only!(
    grid_std::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
)
    i_curr, j_curr = current_pos
    nrows, ncols = size(grid_std)

    w = update_weight(altitude, z_update, transition_k)
    cone_radius = altitude * tan(cone_angle)
    new_std = max(noise_sigma * (1.0 - w), 0.0)

    for i in 1:nrows, j in 1:ncols
        dist = sqrt((i - i_curr)^2 + (j - j_curr)^2)
        if dist <= cone_radius && new_std < grid_std[i, j]
            grid_std[i, j] = new_std
        end
    end

    return nothing
end

# =========================
# MCTS
# =========================

Base.@kwdef struct MCTSConfig
    iterations::Int = 1500
    exploration_c::Float64 = 1.4
    max_rollout_steps::Int = 10_000
    use_max_child_value::Bool = true
end

mutable struct MCTSNode
    state::Tuple{Int,Int,Int}
    std_grid::Matrix{Float64}            # belief std at this decision node (after sensing at this state)
    parent::Int
    incoming_action::Symbol
    incoming_reward::Float64
    children::Dict{Symbol,Int}
    untried_actions::Vector{Symbol}
    visits::Int
    value_sum::Float64
end

function is_terminal_state(s::Tuple{Int,Int,Int})
    return s[3] == 0
end

function sample_terminal_reward(
    mean_grid::Matrix{Float64},
    std_grid::Matrix{Float64},
    next_state::Tuple{Int,Int,Int},
    a::Symbol,
    rng::AbstractRNG,
)
    i, j, _ = next_state
    mu = mean_grid[i, j]
    sigma = max(std_grid[i, j], 0.0)
    landing = sigma == 0.0 ? mu : rand(rng, Normal(mu, sigma))
    return action_penalty(a) + landing
end

function edge_reward_and_child(
    parent::MCTSNode,
    a::Symbol,
    nrows::Int,
    ncols::Int,
    mean_grid::Matrix{Float64},
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    rng::AbstractRNG,
)
    sp = step_next_state(nrows, ncols, parent.state, a)

    r = if sp[3] == 0
        sample_terminal_reward(mean_grid, parent.std_grid, sp, a, rng)
    else
        action_penalty(a)
    end

    child_std = copy(parent.std_grid)
    if sp[3] > 0
        update_with_cone_std_only!(child_std, (sp[1], sp[2]), sp[3], noise_sigma, cone_angle, z_update, transition_k)
    end

    return r, sp, child_std
end

function uct_child_action(nodes::Vector{MCTSNode}, node_idx::Int, c::Float64)
    node = nodes[node_idx]
    best_action = :none
    best_score = -Inf
    parent_visits = max(node.visits, 1)

    for (a, child_idx) in node.children
        child = nodes[child_idx]
        if child.visits == 0
            score = Inf
        else
            q = child.value_sum / child.visits
            score = q + c * sqrt(log(parent_visits) / child.visits)
        end
        if score > best_score
            best_score = score
            best_action = a
        end
    end

    return best_action
end

function rollout_return(
    start_state::Tuple{Int,Int,Int},
    start_std::Matrix{Float64},
    mean_grid::Matrix{Float64},
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    max_rollout_steps::Int,
    rng::AbstractRNG,
)
    nrows, ncols = size(mean_grid)
    s = start_state
    std_grid = copy(start_std)
    total = 0.0
    steps = 0

    while s[3] > 0 && steps < max_rollout_steps
        a = rand(rng, ACTIONS)
        sp = step_next_state(nrows, ncols, s, a)

        if sp[3] == 0
            total += sample_terminal_reward(mean_grid, std_grid, sp, a, rng)
        else
            total += action_penalty(a)
            update_with_cone_std_only!(std_grid, (sp[1], sp[2]), sp[3], noise_sigma, cone_angle, z_update, transition_k)
        end

        s = sp
        steps += 1
    end

    return total
end

function mcts_best_action(
    state::Tuple{Int,Int,Int},
    std_grid::Matrix{Float64},
    mean_grid::Matrix{Float64},
    noise_sigma::Float64,
    cone_angle::Float64;
    z_update::Int,
    transition_k::Float64,
    cfg::MCTSConfig,
    rng::AbstractRNG,
)
    nrows, ncols = size(mean_grid)

    root = MCTSNode(
        state,
        copy(std_grid),
        0,
        :none,
        0.0,
        Dict{Symbol,Int}(),
        copy(ACTIONS),
        0,
        0.0,
    )
    nodes = MCTSNode[root]

    for _ in 1:cfg.iterations
        node_idx = 1
        path = Int[1]
        total = 0.0

        while !isempty(nodes[node_idx].untried_actions) == false && !is_terminal_state(nodes[node_idx].state)
            a = uct_child_action(nodes, node_idx, cfg.exploration_c)
            child_idx = nodes[node_idx].children[a]
            total += nodes[child_idx].incoming_reward
            node_idx = child_idx
            push!(path, node_idx)
        end

        if !is_terminal_state(nodes[node_idx].state) && !isempty(nodes[node_idx].untried_actions)
            a = pop!(nodes[node_idx].untried_actions)
            parent = nodes[node_idx]

            r, sp, child_std = edge_reward_and_child(
                parent,
                a,
                nrows,
                ncols,
                mean_grid,
                noise_sigma,
                cone_angle,
                z_update,
                transition_k,
                rng,
            )

            child = MCTSNode(
                sp,
                child_std,
                node_idx,
                a,
                r,
                Dict{Symbol,Int}(),
                copy(ACTIONS),
                0,
                0.0,
            )
            push!(nodes, child)
            child_idx = length(nodes)
            nodes[node_idx].children[a] = child_idx

            total += r
            node_idx = child_idx
            push!(path, node_idx)
        end

        if !is_terminal_state(nodes[node_idx].state)
            total += rollout_return(
                nodes[node_idx].state,
                nodes[node_idx].std_grid,
                mean_grid,
                noise_sigma,
                cone_angle,
                z_update,
                transition_k,
                cfg.max_rollout_steps,
                rng,
            )
        end

        for idx in path
            nodes[idx].visits += 1
            nodes[idx].value_sum += total
        end
    end

    if isempty(nodes[1].children)
        return :none
    end

    if cfg.use_max_child_value
        best_a = :none
        best_q = -Inf
        for (a, child_idx) in nodes[1].children
            child = nodes[child_idx]
            if child.visits > 0
                q = child.value_sum / child.visits
                if q > best_q
                    best_q = q
                    best_a = a
                end
            end
        end
        return best_a
    else
        best_a = :none
        best_n = -1
        for (a, child_idx) in nodes[1].children
            n = nodes[child_idx].visits
            if n > best_n
                best_n = n
                best_a = a
            end
        end
        return best_a
    end
end

# =========================
# MCTS trajectory planner
# =========================

function plan_with_cone_sensing_mcts(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64}, # kept for API compatibility; intentionally unused by MCTS belief update
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    mcts_cfg::MCTSConfig = MCTSConfig(),
    rng::AbstractRNG = Random.default_rng(),
)
    _ = update_grid

    nrows, ncols = size(initial_mean_grid)
    mean_grid = copy(initial_mean_grid)  # mean remains fixed throughout planning
    grid_std = fill(noise_sigma, nrows, ncols)

    StepT = @NamedTuple{
        state::Tuple{Int,Int,Int},
        action::Symbol,
        reward::Float64,
        next_state::Tuple{Int,Int,Int},
        grid_std_snapshot::Matrix{Float64},
        grid_mean_snapshot::Matrix{Float64}
    }

    trajectory = StepT[]
    current_state = start_state
    total_reward = 0.0

    while current_state[3] > 0
        i, j, z = current_state

        update_with_cone_std_only!(
            grid_std,
            (i, j),
            z,
            noise_sigma,
            cone_angle,
            z_update,
            transition_k,
        )

        action = mcts_best_action(
            current_state,
            grid_std,
            mean_grid,
            noise_sigma,
            cone_angle;
            z_update=z_update,
            transition_k=transition_k,
            cfg=mcts_cfg,
            rng=rng,
        )

        next_state = step_next_state(nrows, ncols, current_state, action)

        r = if next_state[3] == 0
            sample_terminal_reward(mean_grid, grid_std, next_state, action, rng)
        else
            action_penalty(action)
        end

        push!(trajectory, (
            state=current_state,
            action=action,
            reward=r,
            next_state=next_state,
            grid_std_snapshot=copy(grid_std),
            grid_mean_snapshot=copy(mean_grid),
        ))

        total_reward += r
        current_state = next_state
    end

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

function run_batch_mcts(
    n_sims::Int;
    seed::Int = 1234,
    grid_size::Int = 60,
    start_state::Tuple{Int,Int,Int} = (20, 36, 30),
    cone_angle::Float64 = pi / 8,
    noise_sigma::Float64 = 2.0,
    terrain_value_min::Float64 = -10.0,
    terrain_value_range::Float64 = 20.0,
    z_update::Int = 20,
    transition_k::Float64 = 0.0,
    out_csv::String = "batch_results_mcts.csv",
    mcts_cfg::MCTSConfig = MCTSConfig(),
)
    seed_rng = MersenneTwister(seed)

    header = ["run",
              "init_q25","init_med","init_q75",
              "final_q25","final_med","final_q75",
              "landing_value",
              "landing_i","landing_j"]

    rows = Vector{Vector{Any}}(undef, n_sims)

    for run in 1:n_sims
        println("Starting simulation $run / $n_sims...")
        # Keep the same seed consumption pattern as the existing pipeline.
        terrain_seed = rand(seed_rng, 1:10^9)
        noise_seed = rand(seed_rng, 1:10^9)

        initial_mean_grid = generate_terrain_2(
            grid_size;
            seed=terrain_seed,
            value_min=terrain_value_min,
            value_range=terrain_value_range,
        )

        rng_noise = MersenneTwister(noise_seed)
        update_grid = noise_sigma .* randn(rng_noise, grid_size, grid_size)

        # Dedicated planner RNG so MCTS sampling never perturbs terrain/noise seeds.
        planner_rng = MersenneTwister(seed + 10_000 * run)

        traj, total_reward, final_std, final_mean_grid = plan_with_cone_sensing_mcts(
            initial_mean_grid,
            update_grid,
            noise_sigma,
            start_state,
            cone_angle;
            z_update=z_update,
            transition_k=transition_k,
            mcts_cfg=mcts_cfg,
            rng=planner_rng,
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
            land_i, land_j,
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

# Example:
cfg = MCTSConfig(iterations=500, exploration_c=1.2, max_rollout_steps=2000)

#run_batch_mcts(1000; seed=1234, transition_k=0.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k0_sigma0_5.csv")
#run_batch_mcts(1000; seed=1234, transition_k=0.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k0_sigma1_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=0.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k0_sigma2_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=0.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k0_sigma3_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=0.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k0_sigma5_0.csv")

#run_batch_mcts(1000; seed=1234, transition_k=2.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k2_sigma0_5.csv")
#run_batch_mcts(1000; seed=1234, transition_k=2.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k2_sigma1_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=2.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k2_sigma2_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=2.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k2_sigma3_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=2.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k2_sigma5_0.csv")

#run_batch_mcts(1000; seed=1234, transition_k=4.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k4_sigma0_5.csv")
#run_batch_mcts(1000; seed=1234, transition_k=4.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k4_sigma1_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=4.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k4_sigma2_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=4.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k4_sigma3_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=4.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k4_sigma5_0.csv")

#run_batch_mcts(1000; seed=1234, transition_k=7.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k7_sigma0_5.csv")
#run_batch_mcts(1000; seed=1234, transition_k=7.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k7_sigma1_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=7.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k7_sigma2_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=7.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k7_sigma3_0.csv")
#run_batch_mcts(1000; seed=1234, transition_k=7.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k7_sigma5_0.csv")

run_batch_mcts(1000; seed=1234, transition_k=15.0, noise_sigma=0.5, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k15_sigma0_5.csv")
run_batch_mcts(1000; seed=1234, transition_k=15.0, noise_sigma=1.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k15_sigma1_0.csv")
run_batch_mcts(1000; seed=1234, transition_k=15.0, noise_sigma=2.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k15_sigma2_0.csv")
run_batch_mcts(1000; seed=1234, transition_k=15.0, noise_sigma=3.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k15_sigma3_0.csv")
run_batch_mcts(1000; seed=1234, transition_k=15.0, noise_sigma=5.0, terrain_value_min=0.0, terrain_value_range=10.0, z_update=24, mcts_cfg=cfg, out_csv="data_large/batch_mcts_test0_k15_sigma5_0.csv")
