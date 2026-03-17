# =====================================================================
#  Thompson Sampling: target = argmax_{reachable} sample(μ, σ)
#  Requires: environment.jl loaded first
# =====================================================================

function plan_thompson(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    ts_rng::AbstractRNG = Random.GLOBAL_RNG,
)
    nrows, ncols = size(initial_mean_grid)
    mean_grid = copy(initial_mean_grid)
    grid_std  = fill(noise_sigma, nrows, ncols)

    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        update_with_cone!(grid_std, mean_grid, initial_mean_grid, update_grid,
                          (i, j), z, noise_sigma, cone_angle, z_update, transition_k)

        reachable = reachable_indices(nrows, ncols, i, j, z)
        best_cell = (i, j)
        best_sample = -Inf
        for (ri, rj) in reachable
            s = mean_grid[ri, rj] + grid_std[ri, rj] * randn(ts_rng)
            if s > best_sample
                best_sample = s
                best_cell   = (ri, rj)
            end
        end

        action = greedy_step_toward((i, j), best_cell)
        next_state = step_next_state(nrows, ncols, current_state, action)

        r = if next_state[3] == 0
            true_val = initial_mean_grid[next_state[1], next_state[2]] +
                       update_grid[next_state[1], next_state[2]]
            true_val + action_penalty(action)
        else
            action_penalty(action)
        end

        push!(trajectory, (state=current_state, action=action, reward=r,
                           next_state=next_state,
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=best_cell))

        current_state = next_state
    end

    land_i, land_j, _ = trajectory[end].next_state
    landing_value = initial_mean_grid[land_i, land_j] + update_grid[land_i, land_j]
    return trajectory, landing_value, grid_std, mean_grid
end
