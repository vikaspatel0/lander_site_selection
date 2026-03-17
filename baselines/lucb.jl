# =====================================================================
#  LUCB Policy: active set elimination + UCB movement
#
#  Maintains an active set of candidate landing cells.
#  Each step:
#    1. Prune active set to currently reachable cells
#    2. Eliminate cells whose UCB < max LCB (cannot be optimal)
#    3. Move toward best UCB among remaining active cells
# =====================================================================

# Requires: environment.jl loaded first

function plan_lucb(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    α::Float64 = 1.0,
    c::Float64 = 2.0,   # elimination confidence width
)
    nrows, ncols = size(initial_mean_grid)
    mean_grid = copy(initial_mean_grid)
    grid_std  = fill(noise_sigma, nrows, ncols)

    # Active set: all initially reachable cells
    active_set = Set(reachable_indices(nrows, ncols,
                     start_state[1], start_state[2], start_state[3]))

    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        update_with_cone!(grid_std, mean_grid, initial_mean_grid, update_grid,
                          (i, j), z, noise_sigma, cone_angle, z_update, transition_k)

        # Prune active set to currently reachable
        reachable = Set(reachable_indices(nrows, ncols, i, j, z))
        intersect!(active_set, reachable)
        if isempty(active_set)
            active_set = reachable
        end

        # Elimination: remove cells where UCB < max LCB
        max_lcb = -Inf
        for (ri, rj) in active_set
            lcb = mean_grid[ri, rj] - c * grid_std[ri, rj]
            if lcb > max_lcb
                max_lcb = lcb
            end
        end

        to_remove = Tuple{Int,Int}[]
        for (ri, rj) in active_set
            ucb = mean_grid[ri, rj] + c * grid_std[ri, rj]
            if ucb < max_lcb
                push!(to_remove, (ri, rj))
            end
        end
        for cell in to_remove
            delete!(active_set, cell)
        end
        if isempty(active_set)
            active_set = reachable
        end

        # Move toward best UCB among active cells
        best_cell = (i, j)
        best_val  = -Inf
        for (ri, rj) in active_set
            v = mean_grid[ri, rj] + α * grid_std[ri, rj]
            if v > best_val
                best_val  = v
                best_cell = (ri, rj)
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
