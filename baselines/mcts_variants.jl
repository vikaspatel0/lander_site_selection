# =====================================================================
#  MCTS Variants — Four information models × two search types
#
#  Information models (what the rollout/tree uses during planning):
#    V1 (Oracle):          Perfect knowledge of true_terrain
#    V2 (Known dynamics):  Black-box access to observation function (uses true terrain to generate obs)
#    V3 (Frozen belief):   Uses current mean_grid as-is, no simulated observations
#    V4 (Learned dynamics): Estimates per-cell shift from observation history, extrapolates
#
#  Search types:
#    Flat:  Even split of rollouts across actions (MCTS-rollout style)
#    UCT:   Full UCT tree search (MCTS-tree style)
#
#  Outer loop is always the same: real deterministic observations update the belief,
#  then the planner chooses an action using one of the 8 (model × search) combinations.
# =====================================================================

# Requires: environment.jl and mcts_tree.jl loaded first (for MCTSNode, etc.)

# Helper: extract CVaR alpha from version symbol
function cvar_alpha(version::Symbol)
    version == :v3_cvar ? 0.25 : version == :v3_cvar80 ? 0.80 : 0.25
end

function is_cvar_version(version::Symbol)
    version in (:v3_cvar, :v3_cvar80)
end

# ─────────────────────────────────────────────────────────────────────
#  V1: Oracle rollout — uses true_terrain directly
# ─────────────────────────────────────────────────────────────────────

function rollout_v1_oracle(
    true_terrain::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    nrows::Int, ncols::Int,
)
    si, sj, sz = ni, nj, z_next
    while sz > 0
        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz && true_terrain[ii, jj] > best_v
                    best_v = true_terrain[ii, jj]
                    best_ri, best_rj = ii, jj
                end
            end
        end
        act = greedy_step_toward((si, sj), (best_ri, best_rj))
        si, sj = apply_action(si, sj, act, nrows, ncols)
        sz -= 1
    end
    return true_terrain[si, sj]
end

# ─────────────────────────────────────────────────────────────────────
#  V2: Known dynamics rollout — uses observe_and_update! with true terrain
#      Agent treats observation function as black box but gets its true output
# ─────────────────────────────────────────────────────────────────────

function rollout_v2_known_dynamics(
    true_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    noise_sigma::Float64, cone_angle::Float64,
    z_update::Int, transition_k::Float64,
    nrows::Int, ncols::Int,
    rollout_rng::AbstractRNG,
)
    sim_mean = copy(mean_grid)
    sim_std = copy(grid_std)

    si, sj, sz = ni, nj, z_next
    while sz > 0
        # Apply real observation function (deterministic, using true terrain)
        observe_and_update!(sim_mean, sim_std, true_terrain, initial_mean_grid,
                            (si, sj), sz, noise_sigma, cone_angle, z_update, transition_k, rollout_rng)

        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz && sim_mean[ii, jj] > best_v
                    best_v = sim_mean[ii, jj]
                    best_ri, best_rj = ii, jj
                end
            end
        end

        act = greedy_step_toward((si, sj), (best_ri, best_rj))
        si, sj = apply_action(si, sj, act, nrows, ncols)
        sz -= 1
    end
    return true_terrain[si, sj]
end

# ─────────────────────────────────────────────────────────────────────
#  V3: Frozen belief rollout — uses current mean_grid, no updates
# ─────────────────────────────────────────────────────────────────────

function rollout_v3_frozen(
    mean_grid::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    nrows::Int, ncols::Int,
)
    si, sj, sz = ni, nj, z_next
    while sz > 0
        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz && mean_grid[ii, jj] > best_v
                    best_v = mean_grid[ii, jj]
                    best_ri, best_rj = ii, jj
                end
            end
        end
        act = greedy_step_toward((si, sj), (best_ri, best_rj))
        si, sj = apply_action(si, sj, act, nrows, ncols)
        sz -= 1
    end
    return mean_grid[si, sj]
end

# ─────────────────────────────────────────────────────────────────────
#  V3 Risk-aware variants: CVaR and Max-sigma entropic rollouts
#  Same frozen belief, but cell scoring uses risk metrics instead of mean
# ─────────────────────────────────────────────────────────────────────

function rollout_v3_cvar(
    mean_grid::Matrix{Float64},
    std_grid::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    nrows::Int, ncols::Int;
    alpha::Float64 = 0.25,
)
    si, sj, sz = ni, nj, z_next
    while sz > 0
        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz
                    mu = mean_grid[ii, jj]
                    sig = std_grid[ii, jj]
                    v = sig ≈ 0.0 ? mu : mu - sig * pdf(Normal(0,1), quantile(Normal(0,1), alpha)) / alpha
                    if v > best_v
                        best_v = v
                        best_ri, best_rj = ii, jj
                    end
                end
            end
        end
        act = greedy_step_toward((si, sj), (best_ri, best_rj))
        si, sj = apply_action(si, sj, act, nrows, ncols)
        sz -= 1
    end
    # Return CVaR at landing cell — risk-consistent value for tree backpropagation
    mu = mean_grid[si, sj]; sig = std_grid[si, sj]
    return sig ≈ 0.0 ? mu : mu - sig * pdf(Normal(0,1), quantile(Normal(0,1), alpha)) / alpha
end

function rollout_v3_maxsig(
    mean_grid::Matrix{Float64},
    std_grid::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    nrows::Int, ncols::Int;
    beta_risk::Float64 = 0.5,
)
    sigma_ref = maximum(std_grid)
    p = cdf(Normal(0,1), -(beta_risk/2.0) * sigma_ref)

    si, sj, sz = ni, nj, z_next
    while sz > 0
        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz
                    mu = mean_grid[ii, jj]
                    sig = std_grid[ii, jj]
                    v = sig ≈ 0.0 ? mu : quantile(Normal(mu, sig), p)
                    if v > best_v
                        best_v = v
                        best_ri, best_rj = ii, jj
                    end
                end
            end
        end
        act = greedy_step_toward((si, sj), (best_ri, best_rj))
        si, sj = apply_action(si, sj, act, nrows, ncols)
        sz -= 1
    end
    # Return max-sigma entropic value at landing cell
    mu = mean_grid[si, sj]; sig = std_grid[si, sj]
    return sig ≈ 0.0 ? mu : quantile(Normal(mu, sig), p)
end

# ─────────────────────────────────────────────────────────────────────
#  V4: Learned dynamics rollout — estimates per-cell shift from history,
#      extrapolates to lower altitudes using fitted convergence
# ─────────────────────────────────────────────────────────────────────

# Observation history: for each cell, store (altitude, observed_mean) pairs
const ObsHistory = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Float64}}}

# Sliding window size for V4 extrapolation (configurable)
GLOBAL_V4_WINDOW_SIZE = 5

function set_v4_window!(τ::Int)
    global GLOBAL_V4_WINDOW_SIZE = τ
end

function extrapolate_mean(
    initial_mean::Float64,
    history::Vector{Tuple{Int,Float64}},
    target_z::Int,
    z_start::Int,
)
    if isempty(history)
        return initial_mean
    end

    # Sliding window: use only the last τ observations (most recent = lowest altitude)
    τ = GLOBAL_V4_WINDOW_SIZE
    window = length(history) <= τ ? history : history[end-τ+1:end]

    if length(window) == 1
        z_obs, val_obs = window[1]
        shift = val_obs - initial_mean
        progress_obs = (z_start - z_obs) / z_start
        progress_target = (z_start - target_z) / z_start
        if progress_obs > 0 && abs(shift) > 1e-6
            return initial_mean + shift * (progress_target / progress_obs)
        else
            return val_obs
        end
    end

    # Fit line through windowed observations: shift = a * progress + b
    # Using standard linear regression (not forced through origin)
    n = length(window)
    sx = 0.0; sy = 0.0; sxy = 0.0; sx2 = 0.0
    for (z_obs, val_obs) in window
        prog = (z_start - z_obs) / z_start
        shift = val_obs - initial_mean
        sx += prog; sy += shift; sxy += prog * shift; sx2 += prog^2
    end
    denom = n * sx2 - sx^2
    if abs(denom) < 1e-12
        # All at same progress — just return last value
        return window[end][2]
    end
    a = (n * sxy - sx * sy) / denom  # slope
    b = (sy * sx2 - sx * sxy) / denom  # intercept

    progress_target = (z_start - target_z) / z_start
    return initial_mean + a * progress_target + b
end

# Precompute extrapolated grid: for each cell, estimate what it would be at z=0
function build_extrapolated_grid(
    initial_mean_grid::Matrix{Float64},
    mean_grid::Matrix{Float64},
    obs_history::ObsHistory,
    z_start::Int,
)
    nrows, ncols = size(mean_grid)
    extrap = copy(mean_grid)
    for ((ci, cj), hist) in obs_history
        extrap[ci, cj] = extrapolate_mean(initial_mean_grid[ci, cj], hist, 0, z_start)
    end
    return extrap
end

function rollout_v4_learned(
    extrap_grid::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    nrows::Int, ncols::Int,
)
    si, sj, sz = ni, nj, z_next
    while sz > 0
        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz && extrap_grid[ii, jj] > best_v
                    best_v = extrap_grid[ii, jj]
                    best_ri, best_rj = ii, jj
                end
            end
        end
        act = greedy_step_toward((si, sj), (best_ri, best_rj))
        si, sj = apply_action(si, sj, act, nrows, ncols)
        sz -= 1
    end
    return extrap_grid[si, sj]
end

# ─────────────────────────────────────────────────────────────────────
#  Flat MCTS (rollout-based) — all 4 versions
# ─────────────────────────────────────────────────────────────────────

function plan_mcts_flat(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    n_rollouts::Int = 200,
    version::Symbol = :v3,  # :v1, :v2, :v3, :v4
    sample_rng::AbstractRNG = Random.GLOBAL_RNG,
    obs_rng::AbstractRNG = MersenneTwister(0),
    budget::ActionBudget = DEFAULT_BUDGET,
)
    nrows, ncols = size(initial_mean_grid)
    true_terrain = initial_mean_grid .+ update_grid
    mean_grid = copy(initial_mean_grid)
    grid_std = fill(noise_sigma, nrows, ncols)
    z_start = start_state[3]

    # For V4: track observation history
    obs_history = ObsHistory()

    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        # Real observation (deterministic)
        observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid,
                            (i, j), z, noise_sigma, cone_angle, z_update, transition_k, obs_rng)

        # Update observation history for V4
        if version == :v4
            cells = cone_cells((i, j), z, cone_angle, nrows, ncols)
            for (ci, cj) in cells
                if !haskey(obs_history, (ci, cj))
                    obs_history[(ci, cj)] = Tuple{Int,Float64}[]
                end
                push!(obs_history[(ci, cj)], (z, mean_grid[ci, cj]))
            end
        end

        z_next = z - 1

        # Precompute V4 extrapolated grid once per step (not per rollout)
        extrap_grid = version == :v4 ?
            build_extrapolated_grid(initial_mean_grid, mean_grid, obs_history, z_start) :
            mean_grid  # unused placeholder for other versions

        if z_next == 0
            # Last step: pick best adjacent cell by belief mean
            best_act = :none; best_val = -Inf
            for act in ACTIONS
                ni, nj = apply_action(i, j, act, nrows, ncols)
                v = if version == :v1
                    true_terrain[ni, nj]
                elseif version == :v4
                    extrap_grid[ni, nj]
                else
                    mean_grid[ni, nj]
                end
                if v > best_val; best_val = v; best_act = act; end
            end
            ni, nj = apply_action(i, j, best_act, nrows, ncols)
            r = true_terrain[ni, nj] + action_penalty(best_act)
            push!(trajectory, (state=current_state, action=best_act, reward=r,
                               next_state=(ni,nj,0),
                               grid_std_snapshot=copy(grid_std),
                               grid_mean_snapshot=copy(mean_grid),
                               target=(ni,nj)))
            current_state = (ni, nj, 0)
            continue
        end

        # Evaluate each action via rollouts
        t_action = time()
        action_values = zeros(length(ACTIONS))
        action_counts = zeros(Int, length(ACTIONS))

        for (ai, act) in enumerate(ACTIONS)
            ni, nj = apply_action(i, j, act, nrows, ncols)

            for r_idx in 1:n_rollouts
                if !check_time_budget(t_action, budget)
                    break
                end

                val = if version == :v1
                    rollout_v1_oracle(true_terrain, ni, nj, z_next, nrows, ncols)

                elseif version == :v2
                    rollout_rng = MersenneTwister(rand(sample_rng, UInt64))
                    rollout_v2_known_dynamics(true_terrain, initial_mean_grid,
                        mean_grid, grid_std, ni, nj, z_next, noise_sigma, cone_angle,
                        z_update, transition_k, nrows, ncols, rollout_rng)

                elseif version == :v3
                    rollout_v3_frozen(mean_grid, ni, nj, z_next, nrows, ncols)

                elseif version == :v4
                    rollout_v4_learned(extrap_grid, ni, nj, z_next, nrows, ncols)
                else
                    error("Unknown version: $version")
                end

                action_values[ai] += val
                action_counts[ai] += 1
            end
            if action_counts[ai] > 0
                action_values[ai] /= action_counts[ai]
            end
        end

        # Pick best action
        evaluated = findall(c -> c > 0, action_counts)
        best_act = if isempty(evaluated)
            greedy_fallback_action(mean_grid, i, j, z, nrows, ncols)
        else
            ACTIONS[evaluated[argmax(action_values[evaluated])]]
        end

        ni, nj = apply_action(i, j, best_act, nrows, ncols)
        r = z_next == 0 ? true_terrain[ni, nj] + action_penalty(best_act) : action_penalty(best_act)

        push!(trajectory, (state=current_state, action=best_act, reward=r,
                           next_state=(ni, nj, z_next),
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=nothing))
        current_state = (ni, nj, z_next)
    end

    land_i, land_j, _ = trajectory[end].next_state
    return trajectory, true_terrain[land_i, land_j], grid_std, mean_grid
end

# ─────────────────────────────────────────────────────────────────────
#  UCT MCTS — all 4 versions
# ─────────────────────────────────────────────────────────────────────

function uct_rollout_return(
    version::Symbol,
    start_state::Tuple{Int,Int,Int},
    true_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    mean_grid::Matrix{Float64},
    std_grid::Matrix{Float64},
    extrap_grid::Matrix{Float64},
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    max_steps::Int,
    rng::AbstractRNG,
)
    nrows, ncols = size(mean_grid)
    # Pick the evaluation grid based on version
    eval_grid = version == :v1 ? true_terrain :
                version == :v4 ? extrap_grid : mean_grid  # V3/V3 variants use mean_grid

    s = start_state
    total = 0.0
    steps = 0

    while s[3] > 0 && steps < max_steps
        i, j, z = s

        best_ri, best_rj = i, j
        best_v = -Inf

        if version == :v2
            # V2 handled below — needs simulated observation
        elseif is_cvar_version(version)
            # CVaR scoring
            α_cv = cvar_alpha(version)
            for ii in max(1, i-z):min(nrows, i+z)
                for jj in max(1, j-z):min(ncols, j+z)
                    if abs(ii-i) + abs(jj-j) <= z
                        mu = mean_grid[ii, jj]; sig = std_grid[ii, jj]
                        v = sig ≈ 0.0 ? mu : mu - sig * pdf(Normal(0,1), quantile(Normal(0,1), α_cv)) / α_cv
                        if v > best_v; best_v = v; best_ri, best_rj = ii, jj; end
                    end
                end
            end
        elseif version == :v3_maxsig
            # Max-sigma entropic scoring
            sigma_ref = maximum(std_grid)
            p = cdf(Normal(0,1), -(0.5/2.0) * sigma_ref)
            for ii in max(1, i-z):min(nrows, i+z)
                for jj in max(1, j-z):min(ncols, j+z)
                    if abs(ii-i) + abs(jj-j) <= z
                        mu = mean_grid[ii, jj]; sig = std_grid[ii, jj]
                        v = sig ≈ 0.0 ? mu : quantile(Normal(mu, sig), p)
                        if v > best_v; best_v = v; best_ri, best_rj = ii, jj; end
                    end
                end
            end
        else
            # V1, V3, V4: use preselected eval_grid
            for ii in max(1, i-z):min(nrows, i+z)
                for jj in max(1, j-z):min(ncols, j+z)
                    if abs(ii-i) + abs(jj-j) <= z && eval_grid[ii, jj] > best_v
                        best_v = eval_grid[ii, jj]; best_ri, best_rj = ii, jj
                    end
                end
            end
        end
        # V2 handled separately below

        a = if version == :v2
            # Known dynamics: simulate observation, then greedy
            sim_mean = copy(mean_grid); sim_std = copy(std_grid)
            observe_and_update!(sim_mean, sim_std, true_terrain, initial_mean_grid,
                                (i, j), z, noise_sigma, cone_angle, z_update, transition_k, rng)
            best_v2 = -Inf; best_r2i, best_r2j = i, j
            for ii in max(1, i-z):min(nrows, i+z)
                for jj in max(1, j-z):min(ncols, j+z)
                    if abs(ii-i) + abs(jj-j) <= z && sim_mean[ii, jj] > best_v2
                        best_v2 = sim_mean[ii, jj]; best_r2i, best_r2j = ii, jj
                    end
                end
            end
            greedy_step_toward((i, j), (best_r2i, best_r2j))
        else
            greedy_step_toward((i, j), (best_ri, best_rj))
        end

        sp = step_next_state(nrows, ncols, s, a)

        if sp[3] == 0
            # Terminal value depends on version
            landing_val = if version in (:v1, :v2)
                true_terrain[sp[1], sp[2]]
            elseif is_cvar_version(version)
                α_cv = cvar_alpha(version)
                mu = mean_grid[sp[1], sp[2]]; sig = std_grid[sp[1], sp[2]]
                sig ≈ 0.0 ? mu : mu - sig * pdf(Normal(0,1), quantile(Normal(0,1), α_cv)) / α_cv
            elseif version == :v3_maxsig
                mu = mean_grid[sp[1], sp[2]]; sig = std_grid[sp[1], sp[2]]
                sigma_ref = maximum(std_grid)
                p_risk = cdf(Normal(0,1), -(0.5/2.0) * sigma_ref)
                sig ≈ 0.0 ? mu : quantile(Normal(mu, sig), p_risk)
            else
                eval_grid[sp[1], sp[2]]
            end
            total += landing_val + action_penalty(a)
        else
            total += action_penalty(a)
        end

        s = sp
        steps += 1
    end
    return total
end

function plan_mcts_uct(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    version::Symbol = :v3,
    iterations::Int = 700,
    exploration_c::Float64 = 3.0,
    max_rollout_steps::Int = 10_000,
    rng::AbstractRNG = Random.default_rng(),
    obs_rng::AbstractRNG = MersenneTwister(0),
    budget::ActionBudget = DEFAULT_BUDGET,
)
    nrows, ncols = size(initial_mean_grid)
    true_terrain = initial_mean_grid .+ update_grid
    mean_grid = copy(initial_mean_grid)
    grid_std = fill(noise_sigma, nrows, ncols)
    z_start = start_state[3]

    obs_history = ObsHistory()
    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        # Real observation
        observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid,
                            (i, j), z, noise_sigma, cone_angle, z_update, transition_k, obs_rng)

        # Update history for V4
        if version == :v4
            cells = cone_cells((i, j), z, cone_angle, nrows, ncols)
            for (ci, cj) in cells
                if !haskey(obs_history, (ci, cj))
                    obs_history[(ci, cj)] = Tuple{Int,Float64}[]
                end
                push!(obs_history[(ci, cj)], (z, mean_grid[ci, cj]))
            end
        end

        # Precompute V4 extrapolated grid once per step
        extrap_grid = version == :v4 ?
            build_extrapolated_grid(initial_mean_grid, mean_grid, obs_history, z_start) :
            mean_grid

        # UCT tree search
        t_action = time()

        root = MCTSNode(current_state, copy(grid_std), 0, :none, 0.0,
                        Dict{Symbol,Int}(), copy(ACTIONS), 0, 0.0)
        nodes = MCTSNode[root]

        for iter in 1:iterations
            if !check_time_budget(t_action, budget)
                break
            end
            node_idx = 1; path = Int[1]; total = 0.0

            # Selection
            while isempty(nodes[node_idx].untried_actions) && !is_terminal_state(nodes[node_idx].state)
                a = uct_child_action(nodes, node_idx, exploration_c)
                child_idx = nodes[node_idx].children[a]
                total += nodes[child_idx].incoming_reward
                node_idx = child_idx
                push!(path, node_idx)
            end

            # Expansion
            if !is_terminal_state(nodes[node_idx].state) && !isempty(nodes[node_idx].untried_actions)
                a = pop!(nodes[node_idx].untried_actions)
                sp = step_next_state(nrows, ncols, nodes[node_idx].state, a)

                r = if sp[3] == 0
                    # Terminal: risk-consistent value by version
                    if version in (:v1, :v2)
                        true_terrain[sp[1], sp[2]] + action_penalty(a)
                    elseif version == :v4
                        extrap_grid[sp[1], sp[2]] + action_penalty(a)
                    elseif is_cvar_version(version)
                        α_cv = cvar_alpha(version)
                        mu = mean_grid[sp[1], sp[2]]; sig = grid_std[sp[1], sp[2]]
                        (sig ≈ 0.0 ? mu : mu - sig * pdf(Normal(0,1), quantile(Normal(0,1), α_cv)) / α_cv) + action_penalty(a)
                    elseif version == :v3_maxsig
                        mu = mean_grid[sp[1], sp[2]]; sig = grid_std[sp[1], sp[2]]
                        sr = maximum(grid_std); pr = cdf(Normal(0,1), -(0.5/2.0) * sr)
                        (sig ≈ 0.0 ? mu : quantile(Normal(mu, sig), pr)) + action_penalty(a)
                    else
                        mean_grid[sp[1], sp[2]] + action_penalty(a)
                    end
                else
                    action_penalty(a)
                end

                child_std = copy(nodes[node_idx].std_grid)
                if sp[3] > 0
                    bayesian_std_update!(child_std, (sp[1], sp[2]), sp[3],
                                          noise_sigma, cone_angle, z_update, transition_k)
                end

                child = MCTSNode(sp, child_std, node_idx, a, r,
                                 Dict{Symbol,Int}(), copy(ACTIONS), 0, 0.0)
                push!(nodes, child)
                child_idx = length(nodes)
                nodes[node_idx].children[a] = child_idx
                total += r; node_idx = child_idx; push!(path, node_idx)
            end

            # Rollout
            if !is_terminal_state(nodes[node_idx].state)
                total += uct_rollout_return(version, nodes[node_idx].state,
                    true_terrain, initial_mean_grid, mean_grid, nodes[node_idx].std_grid,
                    extrap_grid, noise_sigma, cone_angle, z_update, transition_k,
                    max_rollout_steps, rng)
            end

            # Backprop
            for idx in path
                nodes[idx].visits += 1
                nodes[idx].value_sum += total
            end
        end

        # Pick best action
        action = if isempty(nodes[1].children)
            :none
        else
            best_a = :none; best_q = -Inf
            for (a, child_idx) in nodes[1].children
                child = nodes[child_idx]
                if child.visits > 0
                    q = child.value_sum / child.visits
                    if q > best_q; best_q = q; best_a = a; end
                end
            end
            best_a
        end

        next_state = step_next_state(nrows, ncols, current_state, action)
        r = next_state[3] == 0 ? true_terrain[next_state[1], next_state[2]] + action_penalty(action) : action_penalty(action)

        push!(trajectory, (state=current_state, action=action, reward=r,
                           next_state=next_state,
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=nothing))
        current_state = next_state
    end

    land_i, land_j, _ = trajectory[end].next_state
    return trajectory, true_terrain[land_i, land_j], grid_std, mean_grid
end
