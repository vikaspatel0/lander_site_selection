# =====================================================================
#  Full MCTS Tree Search
#
#  UCT-based tree search with random rollouts.
#  Belief std is updated using expected Bayesian posterior std
#  (deterministic given prior std and obs_std).
#  Landing reward sampled from N(μ, σ²) at terminal states.
# =====================================================================

# Requires: environment.jl loaded first

# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

Base.@kwdef struct MCTSTreeConfig
    iterations::Int = 700
    exploration_c::Float64 = 3.0
    max_rollout_steps::Int = 10_000
    use_max_child_value::Bool = true
    rollout_policy::Symbol = :random   # :random, :greedy, :ucb
    rollout_alpha::Float64 = 1.0       # for :ucb rollout
end

# ─────────────────────────────────────────────────────────────────────
#  Tree node
# ─────────────────────────────────────────────────────────────────────

mutable struct MCTSNode
    state::Tuple{Int,Int,Int}
    std_grid::Matrix{Float64}
    parent::Int
    incoming_action::Symbol
    incoming_reward::Float64
    children::Dict{Symbol,Int}
    untried_actions::Vector{Symbol}
    visits::Int
    value_sum::Float64
end

is_terminal_state(s::Tuple{Int,Int,Int}) = s[3] == 0

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

# ─────────────────────────────────────────────────────────────────────
#  Bayesian expected std update for tree nodes
#
#  Given prior std σ_prior and obs_std, the expected posterior std is:
#    σ_post = 1/sqrt(1/σ_prior² + 1/obs_std²)
#  This is deterministic (does not depend on the actual observation value)
#  and is the correct expected std reduction under the Bayesian model.
# ─────────────────────────────────────────────────────────────────────

function bayesian_std_update!(
    grid_std::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64
)
    nrows, ncols = size(grid_std)
    w = update_weight(altitude, z_update, transition_k)
    obs_std = max(noise_sigma * (1.0 - w), 0.0)

    if altitude == 0
        i, j = current_pos
        grid_std[i, j] = 0.0
        return nothing
    end

    cells = cone_cells(current_pos, altitude, cone_angle, nrows, ncols)
    for (i, j) in cells
        σ_prior = grid_std[i, j]
        if σ_prior <= 0.0
            continue
        end
        if obs_std <= 0.0
            grid_std[i, j] = 0.0
        else
            τ_prior = 1.0 / (σ_prior^2)
            τ_obs = 1.0 / (obs_std^2)
            grid_std[i, j] = 1.0 / sqrt(τ_prior + τ_obs)
        end
    end
    return nothing
end

function edge_reward_and_child(
    parent::MCTSNode,
    a::Symbol,
    nrows::Int, ncols::Int,
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
        bayesian_std_update!(child_std, (sp[1], sp[2]), sp[3],
                              noise_sigma, cone_angle, z_update, transition_k)
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

function rollout_return_random(
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
            bayesian_std_update!(std_grid, (sp[1], sp[2]), sp[3],
                                  noise_sigma, cone_angle, z_update, transition_k)
        end

        s = sp
        steps += 1
    end
    return total
end

function rollout_return_greedy(
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
        i, j, z = s

        # Greedy: move toward best reachable cell by mean
        best_ri, best_rj = i, j
        best_v = -Inf
        for ii in max(1, i-z):min(nrows, i+z)
            for jj in max(1, j-z):min(ncols, j+z)
                if abs(ii-i) + abs(jj-j) <= z && mean_grid[ii, jj] > best_v
                    best_v = mean_grid[ii, jj]
                    best_ri, best_rj = ii, jj
                end
            end
        end

        a = greedy_step_toward((i, j), (best_ri, best_rj))
        sp = step_next_state(nrows, ncols, s, a)

        if sp[3] == 0
            total += sample_terminal_reward(mean_grid, std_grid, sp, a, rng)
        else
            total += action_penalty(a)
            bayesian_std_update!(std_grid, (sp[1], sp[2]), sp[3],
                                  noise_sigma, cone_angle, z_update, transition_k)
        end

        s = sp
        steps += 1
    end
    return total
end

function rollout_return_ucb(
    start_state::Tuple{Int,Int,Int},
    start_std::Matrix{Float64},
    mean_grid::Matrix{Float64},
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    max_rollout_steps::Int,
    rng::AbstractRNG;
    alpha::Float64 = 1.0,
)
    nrows, ncols = size(mean_grid)
    s = start_state
    std_grid = copy(start_std)
    total = 0.0
    steps = 0

    while s[3] > 0 && steps < max_rollout_steps
        i, j, z = s

        best_ri, best_rj = i, j
        best_v = -Inf
        for ii in max(1, i-z):min(nrows, i+z)
            for jj in max(1, j-z):min(ncols, j+z)
                if abs(ii-i) + abs(jj-j) <= z
                    v = mean_grid[ii, jj] + alpha * std_grid[ii, jj]
                    if v > best_v
                        best_v = v
                        best_ri, best_rj = ii, jj
                    end
                end
            end
        end

        a = greedy_step_toward((i, j), (best_ri, best_rj))
        sp = step_next_state(nrows, ncols, s, a)

        if sp[3] == 0
            total += sample_terminal_reward(mean_grid, std_grid, sp, a, rng)
        else
            total += action_penalty(a)
            bayesian_std_update!(std_grid, (sp[1], sp[2]), sp[3],
                                  noise_sigma, cone_angle, z_update, transition_k)
        end

        s = sp
        steps += 1
    end
    return total
end

# ─────────────────────────────────────────────────────────────────────
#  Best action via MCTS tree search
# ─────────────────────────────────────────────────────────────────────

function mcts_tree_best_action(
    state::Tuple{Int,Int,Int},
    std_grid::Matrix{Float64},
    mean_grid::Matrix{Float64},
    noise_sigma::Float64,
    cone_angle::Float64;
    z_update::Int,
    transition_k::Float64,
    cfg::MCTSTreeConfig,
    rng::AbstractRNG,
    budget::ActionBudget = DEFAULT_BUDGET,
    t_action_start::Float64 = time(),
)
    nrows, ncols = size(mean_grid)

    root = MCTSNode(state, copy(std_grid), 0, :none, 0.0,
                    Dict{Symbol,Int}(), copy(ACTIONS), 0, 0.0)
    nodes = MCTSNode[root]

    for iter in 1:cfg.iterations
        if !check_time_budget(t_action_start, budget)
            break
        end
        node_idx = 1
        path = Int[1]
        total = 0.0

        # Selection
        while isempty(nodes[node_idx].untried_actions) && !is_terminal_state(nodes[node_idx].state)
            a = uct_child_action(nodes, node_idx, cfg.exploration_c)
            child_idx = nodes[node_idx].children[a]
            total += nodes[child_idx].incoming_reward
            node_idx = child_idx
            push!(path, node_idx)
        end

        # Expansion
        if !is_terminal_state(nodes[node_idx].state) && !isempty(nodes[node_idx].untried_actions)
            a = pop!(nodes[node_idx].untried_actions)
            r, sp, child_std = edge_reward_and_child(
                nodes[node_idx], a, nrows, ncols, mean_grid,
                noise_sigma, cone_angle, z_update, transition_k, rng)

            child = MCTSNode(sp, child_std, node_idx, a, r,
                             Dict{Symbol,Int}(), copy(ACTIONS), 0, 0.0)
            push!(nodes, child)
            child_idx = length(nodes)
            nodes[node_idx].children[a] = child_idx

            total += r
            node_idx = child_idx
            push!(path, node_idx)
        end

        # Rollout
        if !is_terminal_state(nodes[node_idx].state)
            rollout_fn = if cfg.rollout_policy == :greedy
                (s, std, mg, ns, ca, zu, tk, ms, r) ->
                    rollout_return_greedy(s, std, mg, ns, ca, zu, tk, ms, r)
            elseif cfg.rollout_policy == :ucb
                (s, std, mg, ns, ca, zu, tk, ms, r) ->
                    rollout_return_ucb(s, std, mg, ns, ca, zu, tk, ms, r; alpha=cfg.rollout_alpha)
            else  # :random
                rollout_return_random
            end
            total += rollout_fn(
                nodes[node_idx].state, nodes[node_idx].std_grid,
                mean_grid, noise_sigma, cone_angle,
                z_update, transition_k, cfg.max_rollout_steps, rng)
        end

        # Backprop
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

# ─────────────────────────────────────────────────────────────────────
#  Full trajectory planner using MCTS tree search
# ─────────────────────────────────────────────────────────────────────

function plan_mcts_tree(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    mcts_cfg::MCTSTreeConfig = MCTSTreeConfig(),
    rng::AbstractRNG = Random.default_rng(),
    obs_rng::AbstractRNG = MersenneTwister(0),
    budget::ActionBudget = DEFAULT_BUDGET,
)
    nrows, ncols = size(initial_mean_grid)
    true_terrain = initial_mean_grid .+ update_grid
    mean_grid = copy(initial_mean_grid)
    grid_std = fill(noise_sigma, nrows, ncols)

    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        # Real observation: Bayesian update from true terrain
        observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid,
                            (i, j), z, noise_sigma, cone_angle, z_update, transition_k, obs_rng)

        t_action = time()
        action = mcts_tree_best_action(
            current_state, grid_std, mean_grid, noise_sigma, cone_angle;
            z_update=z_update, transition_k=transition_k,
            cfg=mcts_cfg, rng=rng,
            budget=budget, t_action_start=t_action)

        next_state = step_next_state(nrows, ncols, current_state, action)

        r = if next_state[3] == 0
            true_val = true_terrain[next_state[1], next_state[2]]
            true_val + action_penalty(action)
        else
            action_penalty(action)
        end

        push!(trajectory, (state=current_state, action=action, reward=r,
                           next_state=next_state,
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=nothing))

        current_state = next_state
    end

    land_i, land_j, _ = trajectory[end].next_state
    landing_value = true_terrain[land_i, land_j]
    return trajectory, landing_value, grid_std, mean_grid
end
