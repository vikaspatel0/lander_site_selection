# =====================================================================
#  MCTS Depth-1 with Monte Carlo Rollouts
#
#  At each step:
#    1. Update belief via cone observation
#    2. For each of 5 actions, run n_rollouts:
#       a. Sample terrain from current belief N(μ, σ²)
#       b. Simulate continuation with chosen rollout policy
#    3. Take action with highest average landing value
#
#  Rollout policies: :greedy (best μ), :ucb (μ + α·σ), :coneinfo (μ + α·Σσ_cone)
# =====================================================================

# Requires: environment.jl loaded first

# ─────────────────────────────────────────────────────────────────────
#  Rollout policies (operate on sampled terrain with simulated sensing)
# ─────────────────────────────────────────────────────────────────────

function rollout_greedy(
    sampled_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    update_grid_sampled::Matrix{Float64},
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    noise_sigma::Float64, cone_angle::Float64,
    z_update::Int, transition_k::Float64,
    nrows::Int, ncols::Int
)
    sim_mean = copy(mean_grid)
    sim_std  = copy(grid_std)

    si, sj, sz = ni, nj, z_next
    while sz > 0
        update_with_cone!(sim_std, sim_mean, initial_mean_grid, update_grid_sampled,
                          (si, sj), sz, noise_sigma, cone_angle, z_update, transition_k)

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
    return sampled_terrain[si, sj]
end

function rollout_ucb(
    sampled_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    update_grid_sampled::Matrix{Float64},
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    noise_sigma::Float64, cone_angle::Float64,
    z_update::Int, transition_k::Float64,
    nrows::Int, ncols::Int;
    alpha::Float64 = 1.0
)
    sim_mean = copy(mean_grid)
    sim_std  = copy(grid_std)

    si, sj, sz = ni, nj, z_next
    while sz > 0
        update_with_cone!(sim_std, sim_mean, initial_mean_grid, update_grid_sampled,
                          (si, sj), sz, noise_sigma, cone_angle, z_update, transition_k)

        best_ri, best_rj = si, sj
        best_v = -Inf
        for ii in max(1, si-sz):min(nrows, si+sz)
            for jj in max(1, sj-sz):min(ncols, sj+sz)
                if abs(ii-si) + abs(jj-sj) <= sz
                    v = sim_mean[ii, jj] + alpha * sim_std[ii, jj]
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
    return sampled_terrain[si, sj]
end

function rollout_coneinfo(
    sampled_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    update_grid_sampled::Matrix{Float64},
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    ni::Int, nj::Int, z_next::Int,
    noise_sigma::Float64, cone_angle::Float64,
    z_update::Int, transition_k::Float64,
    nrows::Int, ncols::Int;
    alpha::Float64 = 1.0
)
    sim_mean = copy(mean_grid)
    sim_std  = copy(grid_std)

    si, sj, sz = ni, nj, z_next
    while sz > 0
        update_with_cone!(sim_std, sim_mean, initial_mean_grid, update_grid_sampled,
                          (si, sj), sz, noise_sigma, cone_angle, z_update, transition_k)

        best_act = :none
        best_score = -Inf
        for act in ACTIONS
            ai, aj = apply_action(si, sj, act, nrows, ncols)
            next_alt = sz - 1

            # Best reachable mean from (ai, aj) at altitude next_alt
            best_mu = -Inf
            if next_alt > 0
                for ii in max(1, ai-next_alt):min(nrows, ai+next_alt)
                    for jj in max(1, aj-next_alt):min(ncols, aj+next_alt)
                        if abs(ii-ai) + abs(jj-aj) <= next_alt && sim_mean[ii, jj] > best_mu
                            best_mu = sim_mean[ii, jj]
                        end
                    end
                end
            else
                best_mu = sim_mean[ai, aj]
            end

            info = next_alt > 0 ?
                cone_info_gain(ai, aj, next_alt, sim_std, cone_angle, nrows, ncols) : 0.0

            score = best_mu + alpha * info
            if score > best_score
                best_score = score
                best_act = act
            end
        end

        si, sj = apply_action(si, sj, best_act, nrows, ncols)
        sz -= 1
    end
    return sampled_terrain[si, sj]
end

# ─────────────────────────────────────────────────────────────────────
#  Main MCTS d=1 planner
# ─────────────────────────────────────────────────────────────────────

function plan_mcts_rollout(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    n_rollouts::Int = 200,
    rollout_policy::Symbol = :greedy,
    rollout_alpha::Float64 = 1.0,
    sample_rng::AbstractRNG = Random.GLOBAL_RNG,
    budget::ActionBudget = DEFAULT_BUDGET,
)
    nrows, ncols = size(initial_mean_grid)
    mean_grid = copy(initial_mean_grid)
    grid_std  = fill(noise_sigma, nrows, ncols)

    trajectory = TrajectoryStep[]
    current_state = start_state
    sampled_terrain = Matrix{Float64}(undef, nrows, ncols)
    sampled_update  = Matrix{Float64}(undef, nrows, ncols)

    while current_state[3] > 0
        i, j, z = current_state

        # 1. Real observation
        update_with_cone!(grid_std, mean_grid, initial_mean_grid, update_grid,
                          (i, j), z, noise_sigma, cone_angle, z_update, transition_k)

        z_next = z - 1

        if z_next == 0
            # Last step: pick best adjacent cell by belief mean
            best_act = :none
            best_val = -Inf
            for act in ACTIONS
                ni, nj = apply_action(i, j, act, nrows, ncols)
                if mean_grid[ni, nj] > best_val
                    best_val = mean_grid[ni, nj]
                    best_act = act
                end
            end
            ni, nj = apply_action(i, j, best_act, nrows, ncols)
            true_val = initial_mean_grid[ni, nj] + update_grid[ni, nj]
            r = true_val + action_penalty(best_act)
            push!(trajectory, (state=current_state, action=best_act, reward=r,
                               next_state=(ni,nj,0),
                               grid_std_snapshot=copy(grid_std),
                               grid_mean_snapshot=copy(mean_grid),
                               target=(ni,nj)))
            current_state = (ni, nj, 0)
            continue
        end

        # 2. Evaluate each action via rollouts (with time budget)
        t_action = time()
        action_values = zeros(length(ACTIONS))
        action_counts = zeros(Int, length(ACTIONS))
        timed_out = false

        for (ai, act) in enumerate(ACTIONS)
            ni, nj = apply_action(i, j, act, nrows, ncols)

            for r_idx in 1:n_rollouts
                if !check_time_budget(t_action, budget)
                    timed_out = true
                    break
                end

                # Sample terrain from current belief
                for ci in 1:nrows, cj in 1:ncols
                    sampled_terrain[ci, cj] = mean_grid[ci, cj] + grid_std[ci, cj] * randn(sample_rng)
                    sampled_update[ci, cj]  = sampled_terrain[ci, cj] - initial_mean_grid[ci, cj]
                end

                if rollout_policy == :ucb
                    action_values[ai] += rollout_ucb(
                        sampled_terrain, initial_mean_grid, sampled_update,
                        mean_grid, grid_std,
                        ni, nj, z_next, noise_sigma, cone_angle,
                        z_update, transition_k, nrows, ncols;
                        alpha=rollout_alpha)
                elseif rollout_policy == :coneinfo
                    action_values[ai] += rollout_coneinfo(
                        sampled_terrain, initial_mean_grid, sampled_update,
                        mean_grid, grid_std,
                        ni, nj, z_next, noise_sigma, cone_angle,
                        z_update, transition_k, nrows, ncols;
                        alpha=rollout_alpha)
                else  # :greedy
                    action_values[ai] += rollout_greedy(
                        sampled_terrain, initial_mean_grid, sampled_update,
                        mean_grid, grid_std,
                        ni, nj, z_next, noise_sigma, cone_angle,
                        z_update, transition_k, nrows, ncols)
                end
                action_counts[ai] += 1
            end
            if action_counts[ai] > 0
                action_values[ai] /= action_counts[ai]
            end
            timed_out && break
        end

        # 3. Take best action (fallback to greedy if no rollouts completed for some actions)
        best_act = if timed_out && maximum(action_counts) == 0
            greedy_fallback_action(mean_grid, i, j, z, nrows, ncols)
        else
            # Among actions with at least 1 rollout, pick best
            evaluated = findall(c -> c > 0, action_counts)
            isempty(evaluated) ?
                greedy_fallback_action(mean_grid, i, j, z, nrows, ncols) :
                ACTIONS[evaluated[argmax(action_values[evaluated])]]
        end
        ni, nj = apply_action(i, j, best_act, nrows, ncols)

        r = if z_next == 0
            true_val = initial_mean_grid[ni, nj] + update_grid[ni, nj]
            true_val + action_penalty(best_act)
        else
            action_penalty(best_act)
        end

        push!(trajectory, (state=current_state, action=best_act, reward=r,
                           next_state=(ni, nj, z_next),
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=nothing))

        current_state = (ni, nj, z_next)
    end

    land_i, land_j, _ = trajectory[end].next_state
    landing_value = initial_mean_grid[land_i, land_j] + update_grid[land_i, land_j]
    return trajectory, landing_value, grid_std, mean_grid
end
