if !@isdefined(_ENV_LOADED)
const _ENV_LOADED = true

# =====================================================================
#  Shared environment for lander site selection baselines
#
#  Deterministic observation model:
#    true_terrain = initial_mean_grid + update_grid
#    At altitude z, the agent observes within a cone:
#      w = update_weight(z, z_update, transition_k)
#      mean_grid[i,j] = initial_mean_grid[i,j] + w * update_grid[i,j]
#      grid_std[i,j]  = noise_sigma * (1 - w)
#
#  All baselines include this file and build on these primitives.
# =====================================================================

using Distributions
using Random
using Statistics
using DelimitedFiles

# ─────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────

const ACTIONS = [:up, :down, :left, :right, :none]

# ─────────────────────────────────────────────────────────────────────
#  Terrain generation
# ─────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────
#  Movement / reachability
# ─────────────────────────────────────────────────────────────────────

manhattan(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = abs(a[1]-b[1]) + abs(a[2]-b[2])

function step_next_state(nrows::Int, ncols::Int, s::Tuple{Int,Int,Int}, a::Symbol)
    i, j, z = s
    z_next = max(z - 1, 0)
    if a == :up
        return (max(i-1, 1), j, z_next)
    elseif a == :down
        return (min(i+1, nrows), j, z_next)
    elseif a == :left
        return (i, max(j-1, 1), z_next)
    elseif a == :right
        return (i, min(j+1, ncols), z_next)
    else
        return (i, j, z_next)
    end
end

function apply_action(i::Int, j::Int, a::Symbol, nrows::Int, ncols::Int)
    if a == :up;     return (max(i-1,1), j)
    elseif a == :down;  return (min(i+1,nrows), j)
    elseif a == :left;  return (i, max(j-1,1))
    elseif a == :right; return (i, min(j+1,ncols))
    else; return (i, j)
    end
end

function greedy_step_toward(pos::Tuple{Int,Int}, target::Tuple{Int,Int})
    i, j = pos
    ti, tj = target
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

function reachable_indices(nrows::Int, ncols::Int, i::Int, j::Int, steps::Int)
    inds = Tuple{Int,Int}[]
    for ii in max(1, i-steps):min(nrows, i+steps)
        di = abs(ii - i)
        hr = steps - di
        for jj in max(1, j-hr):min(ncols, j+hr)
            push!(inds, (ii, jj))
        end
    end
    return inds
end

action_penalty(a::Symbol) = (a == :none) ? 0.0 : -0.01

# ─────────────────────────────────────────────────────────────────────
#  Cone sensing (deterministic observation model)
# ─────────────────────────────────────────────────────────────────────

function update_weight(z::Int, z_update::Int, transition_k::Float64)
    if z_update <= 0;  return 0.0; end
    if z <= 0;         return 1.0; end
    if z >= z_update;  return 0.0; end
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
    new_std = max(noise_sigma * (1.0 - w), 0.0)

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

function update_with_cone_stochastic!(
    grid_std::Matrix{Float64},
    mean_grid::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    rng::AbstractRNG
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
                alt_obsv_mean = initial_mean_grid[i, j] + w * update_grid[i, j]
                mean_grid[i, j] = alt_obsv_mean + new_std * randn(rng)
            end
        end
    end

    if altitude == 0
        grid_std[i_curr, j_curr] = 0.0
        mean_grid[i_curr, j_curr] = initial_mean_grid[i_curr, j_curr] + update_grid[i_curr, j_curr]
    end
    return nothing
end

function update_with_cone_std_only!(
    grid_std::Matrix{Float64},
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
    new_std = max(noise_sigma * (1.0 - w), 0.0)

    for i in 1:nrows, j in 1:ncols
        dist = sqrt((i - i_curr)^2 + (j - j_curr)^2)
        if dist <= cone_radius && new_std < grid_std[i, j]
            grid_std[i, j] = new_std
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
#  Trajectory step type
# ─────────────────────────────────────────────────────────────────────

const TrajectoryStep = @NamedTuple{
    state::Tuple{Int,Int,Int},
    action::Symbol,
    reward::Float64,
    next_state::Tuple{Int,Int,Int},
    grid_std_snapshot::Matrix{Float64},
    grid_mean_snapshot::Matrix{Float64},
    target::Union{Nothing,Tuple{Int,Int}}
}

# ─────────────────────────────────────────────────────────────────────
#  Percentile helpers
# ─────────────────────────────────────────────────────────────────────

percentiles(v::AbstractVector{<:Real}) = (
    q25 = quantile(v, 0.25),
    q50 = quantile(v, 0.50),
    q75 = quantile(v, 0.75),
)

# ─────────────────────────────────────────────────────────────────────
#  Cone information gain
# ─────────────────────────────────────────────────────────────────────

function cone_info_gain(si::Int, sj::Int, sz::Int,
                        grid_std::Matrix{Float64},
                        cone_angle::Float64,
                        nrows::Int, ncols::Int)
    cone_radius = sz * tan(cone_angle)
    total_sigma = 0.0
    for ii in max(1, Int(floor(si - cone_radius))):min(nrows, Int(ceil(si + cone_radius)))
        for jj in max(1, Int(floor(sj - cone_radius))):min(ncols, Int(ceil(sj + cone_radius)))
            dist = sqrt(Float64((ii - si)^2 + (jj - sj)^2))
            if dist <= cone_radius
                total_sigma += grid_std[ii, jj]
            end
        end
    end
    return total_sigma
end

# ─────────────────────────────────────────────────────────────────────
#  Action budget: time and memory limits per decision
# ─────────────────────────────────────────────────────────────────────

Base.@kwdef struct ActionBudget
    time_limit_s::Float64 = 30.0    # seconds per action decision
    memory_limit_bytes::Int = 256 * 1024 * 1024  # 256 MB per action decision
end

const DEFAULT_BUDGET = ActionBudget()

"""
    check_time_budget(t_start, budget) -> Bool

Returns true if still within time budget.
"""
check_time_budget(t_start::Float64, budget::ActionBudget) =
    (time() - t_start) < budget.time_limit_s

"""
    remaining_time(t_start, budget) -> Float64

Seconds remaining in the time budget.
"""
remaining_time(t_start::Float64, budget::ActionBudget) =
    max(0.0, budget.time_limit_s - (time() - t_start))

"""
    greedy_fallback_action(mean_grid, i, j, z, nrows, ncols) -> Symbol

Emergency fallback: move toward the reachable cell with highest belief mean.
Used when a policy exceeds its time budget.
"""
function greedy_fallback_action(mean_grid::Matrix{Float64},
                                 i::Int, j::Int, z::Int,
                                 nrows::Int, ncols::Int)
    best_cell = (i, j)
    best_val  = -Inf
    for ii in max(1, i-z):min(nrows, i+z)
        di = abs(ii - i)
        for jj in max(1, j-di):min(ncols, j+di)  # fix: should be j ± (z-di)
            if abs(ii-i) + abs(jj-j) <= z && mean_grid[ii, jj] > best_val
                best_val  = mean_grid[ii, jj]
                best_cell = (ii, jj)
            end
        end
    end
    return greedy_step_toward((i, j), best_cell)
end

end  # _ENV_LOADED guard
