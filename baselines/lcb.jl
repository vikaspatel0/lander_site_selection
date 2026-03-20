# =====================================================================
#  LCB Policy: target = argmax_{reachable} (μ - α·σ)   [risk-averse]
#  Requires: environment.jl loaded first
# =====================================================================

function plan_lcb(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    α::Float64 = 1.0,
    obs_rng::AbstractRNG = MersenneTwister(0),
)
    nrows, ncols = size(initial_mean_grid)
    true_terrain = initial_mean_grid .+ update_grid
    mean_grid = copy(initial_mean_grid)
    grid_std  = fill(noise_sigma, nrows, ncols)

    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid,
                            (i, j), z, noise_sigma, cone_angle, z_update, transition_k, obs_rng)

        reachable = reachable_indices(nrows, ncols, i, j, z)
        best_cell = (i, j)
        best_val  = -Inf
        for (ri, rj) in reachable
            v = mean_grid[ri, rj] - α * grid_std[ri, rj]
            if v > best_val
                best_val  = v
                best_cell = (ri, rj)
            end
        end

        action = greedy_step_toward((i, j), best_cell)
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
                           target=best_cell))

        current_state = next_state
    end

    land_i, land_j, _ = trajectory[end].next_state
    landing_value = true_terrain[land_i, land_j]
    return trajectory, landing_value, grid_std, mean_grid
end
