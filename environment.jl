if !@isdefined(_ENV_LOADED)
const _ENV_LOADED = true

# =====================================================================
#  Shared environment for lander site selection baselines
#
#  Observation model (environment/agent separation):
#
#  ENVIRONMENT (has ground truth):
#    true_terrain[i,j] = initial_mean_grid[i,j] + update_grid[i,j]
#    At altitude z, cells in the cone produce sensor readings:
#      obs_std = noise_sigma * (1 - w),  w = update_weight(z, ...)
#      y[i,j] = true_terrain[i,j] + N(0, obs_std²)
#    → generate_cone_observation(true_terrain, ...)
#
#  AGENT (no access to ground truth):
#    Receives sensor readings y[i,j] and updates belief via
#    Bayesian conjugate normal-normal update:
#      τ_post = τ_prior + τ_obs
#      μ_post = (τ_prior·μ_prior + τ_obs·y) / τ_post
#    → bayesian_update!(mean_grid, grid_std, observations, obs_std)
#
#  All baselines include this file and build on these primitives.
# =====================================================================

using Distributions
using Random

# ─────────────────────────────────────────────────────────────────────
#  Global update mode setting (togglable)
#  :wholesale         — replace mean/std
#  :bayesian          — conjugate normal-normal update
#  :bayesian_decay    — Bayesian with exponential precision decay
#  :altitude_weighted — Bayesian with altitude-informed blending
# ─────────────────────────────────────────────────────────────────────
GLOBAL_UPDATE_MODE = :altitude_weighted
GLOBAL_DECAY_LAMBDA = 0.6
using Statistics
using DelimitedFiles

# ─────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────

# Movement mode: :standard (5 actions, 1 cell) or :extended (9 actions, 1 or 2 cells)
GLOBAL_MOVEMENT_MODE = :standard

const ACTIONS_STANDARD = [:up, :down, :left, :right, :none]
const ACTIONS_EXTENDED = [:up, :down, :left, :right, :up2, :down2, :left2, :right2, :none]

ACTIONS = ACTIONS_STANDARD

function set_movement_mode!(mode::Symbol)
    global GLOBAL_MOVEMENT_MODE = mode
    global ACTIONS = mode == :extended ? ACTIONS_EXTENDED : ACTIONS_STANDARD
end

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
    if a == :up;       return (max(i-1, 1), j, z_next)
    elseif a == :down;  return (min(i+1, nrows), j, z_next)
    elseif a == :left;  return (i, max(j-1, 1), z_next)
    elseif a == :right; return (i, min(j+1, ncols), z_next)
    elseif a == :up2;   return (max(i-2, 1), j, z_next)
    elseif a == :down2; return (min(i+2, nrows), j, z_next)
    elseif a == :left2; return (i, max(j-2, 1), z_next)
    elseif a == :right2;return (i, min(j+2, ncols), z_next)
    else;              return (i, j, z_next)
    end
end

function apply_action(i::Int, j::Int, a::Symbol, nrows::Int, ncols::Int)
    if a == :up;       return (max(i-1,1), j)
    elseif a == :down;  return (min(i+1,nrows), j)
    elseif a == :left;  return (i, max(j-1,1))
    elseif a == :right; return (i, min(j+1,ncols))
    elseif a == :up2;   return (max(i-2,1), j)
    elseif a == :down2; return (min(i+2,nrows), j)
    elseif a == :left2; return (i, max(j-2,1))
    elseif a == :right2;return (i, min(j+2,ncols))
    else;              return (i, j)
    end
end

function greedy_step_toward(pos::Tuple{Int,Int}, target::Tuple{Int,Int})
    i, j = pos
    ti, tj = target
    di = ti - i
    dj = tj - j
    if GLOBAL_MOVEMENT_MODE == :extended
        # Prefer 2-cell moves when distance >= 2
        if abs(di) >= abs(dj) && di != 0
            return abs(di) >= 2 ? (di > 0 ? :down2 : :up2) : (di > 0 ? :down : :up)
        elseif dj != 0
            return abs(dj) >= 2 ? (dj > 0 ? :right2 : :left2) : (dj > 0 ? :right : :left)
        else
            return :none
        end
    else
        if abs(di) >= abs(dj) && di != 0
            return di > 0 ? :down : :up
        elseif dj != 0
            return dj > 0 ? :right : :left
        else
            return :none
        end
    end
end

function reachable_indices(nrows::Int, ncols::Int, i::Int, j::Int, steps::Int)
    # With extended movement (2 cells/step), max Manhattan distance = 2*steps
    max_dist = GLOBAL_MOVEMENT_MODE == :extended ? 2 * steps : steps
    inds = Tuple{Int,Int}[]
    for ii in max(1, i-max_dist):min(nrows, i+max_dist)
        di = abs(ii - i)
        hr = max_dist - di
        for jj in max(1, j-hr):min(ncols, j+hr)
            push!(inds, (ii, jj))
        end
    end
    return inds
end

action_penalty(a::Symbol) = (a == :none) ? 0.0 : -0.01

# ─────────────────────────────────────────────────────────────────────
#  Observation weight (shared by all observation models)
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

# ─────────────────────────────────────────────────────────────────────
#  Cone geometry helper: returns indices of cells inside the cone
# ─────────────────────────────────────────────────────────────────────

function cone_cells(current_pos::Tuple{Int,Int}, altitude::Int,
                    cone_angle::Float64, nrows::Int, ncols::Int)
    i_curr, j_curr = current_pos
    cone_radius = altitude * tan(cone_angle)
    cells = Tuple{Int,Int}[]
    r_int = Int(ceil(cone_radius))
    for i in max(1, i_curr - r_int):min(nrows, i_curr + r_int)
        for j in max(1, j_curr - r_int):min(ncols, j_curr + r_int)
            if sqrt((i - i_curr)^2 + (j - j_curr)^2) <= cone_radius
                push!(cells, (i, j))
            end
        end
    end
    return cells
end

# ─────────────────────────────────────────────────────────────────────
#  ENVIRONMENT SIDE: generate sensor observations
#
#  These functions use the TRUE terrain to produce what the sensor
#  would return. The agent never calls these directly — only the
#  simulation loop does.
# ─────────────────────────────────────────────────────────────────────

"""
    generate_cone_observation(true_terrain, initial_mean_grid, current_pos,
                              altitude, noise_sigma, cone_angle, z_update,
                              transition_k, rng)

Environment-side function. Generates noisy sensor readings for all cells
in the observation cone. Returns a Dict mapping (i,j) => observed_value.

The observation model:
  signal[i,j] = (1-w) * initial_mean[i,j] + w * true_terrain[i,j]
  obs[i,j]    = signal[i,j] + obs_std * randn
where obs_std = noise_sigma * (1 - w).

At high altitude (w≈0): obs ≈ initial_mean + noise (uninformative)
At low altitude (w≈1): obs ≈ true_terrain (precise)
At altitude 0: obs = true_terrain exactly.
"""
function generate_cone_observation(
    true_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    rng::AbstractRNG
)
    nrows, ncols = size(true_terrain)
    w = update_weight(altitude, z_update, transition_k)
    obs_std = max(noise_sigma * (1.0 - w), 0.0)

    observations = Dict{Tuple{Int,Int}, Float64}()

    if altitude == 0
        i, j = current_pos
        observations[(i, j)] = true_terrain[i, j]
    else
        for (i, j) in cone_cells(current_pos, altitude, cone_angle, nrows, ncols)
            signal = (1.0 - w) * initial_mean_grid[i, j] + w * true_terrain[i, j]
            if obs_std > 0.0
                observations[(i, j)] = signal + obs_std * randn(rng)
            else
                observations[(i, j)] = signal
            end
        end
    end

    return observations, obs_std
end

# ─────────────────────────────────────────────────────────────────────
#  AGENT SIDE: Bayesian belief update from sensor observations
#
#  The agent receives raw sensor readings and updates its belief.
#  It does NOT have access to the true terrain.
# ─────────────────────────────────────────────────────────────────────

"""
    bayesian_update!(mean_grid, grid_std, observations, obs_std; mode)

Agent-side function. Updates the belief (mean_grid, grid_std) given
sensor observations.

Modes:
  :wholesale  — replace mean with obs, std with obs_std (if obs_std < σ_prior).
  :bayesian   — Bayesian conjugate normal-normal update (accumulates information).
  :bayesian_decay — Bayesian with exponential precision decay (lambda parameter).
  :altitude_weighted — Bayesian with altitude-informed blending:
                τ_post = (1-w)·τ_prior + w·τ_obs, where w = observation weight from altitude.
                At high altitude (w≈0): trust prior, ignore observation.
                At low altitude (w≈1): full Bayesian update.
"""
function bayesian_update!(
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    observations::Dict{Tuple{Int,Int}, Float64},
    obs_std::Float64;
    mode::Symbol = :wholesale,
    lambda::Float64 = 0.5,  # used for :bayesian_decay
    obs_weight::Float64 = 1.0,  # used for :altitude_weighted (w from update_weight)
)
    for ((i, j), y) in observations
        σ_prior = grid_std[i, j]
        if σ_prior <= 0.0
            continue  # already know this cell exactly
        end

        if obs_std <= 0.0
            # exact observation (all modes agree)
            mean_grid[i, j] = y
            grid_std[i, j] = 0.0
        elseif mode == :wholesale
            if obs_std < σ_prior
                mean_grid[i, j] = y
                grid_std[i, j] = obs_std
            end
        elseif mode == :bayesian
            τ_prior = 1.0 / (σ_prior^2)
            τ_obs = 1.0 / (obs_std^2)
            τ_post = τ_prior + τ_obs
            mean_grid[i, j] = (τ_prior * mean_grid[i, j] + τ_obs * y) / τ_post
            grid_std[i, j] = 1.0 / sqrt(τ_post)
        elseif mode == :bayesian_decay
            τ_prior = lambda / (σ_prior^2)
            τ_obs = 1.0 / (obs_std^2)
            τ_post = τ_prior + τ_obs
            mean_grid[i, j] = (τ_prior * mean_grid[i, j] + τ_obs * y) / τ_post
            grid_std[i, j] = 1.0 / sqrt(τ_post)
        elseif mode == :altitude_weighted
            w = obs_weight
            τ_prior = 1.0 / (σ_prior^2)
            τ_obs = 1.0 / (obs_std^2)
            τ_post = (1.0 - w) * τ_prior + w * τ_obs
            if τ_post > 0.0
                mean_grid[i, j] = ((1.0 - w) * τ_prior * mean_grid[i, j] + w * τ_obs * y) / τ_post
                grid_std[i, j] = 1.0 / sqrt(τ_post)
            end
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
#  SIMULATION CONVENIENCE: observe + update in one call
# ─────────────────────────────────────────────────────────────────────

"""
    observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid,
                        current_pos, altitude, noise_sigma, cone_angle,
                        z_update, transition_k, rng)

Convenience function for the simulation loop. Generates observations
from the true terrain (environment side), then updates the agent's
belief (agent side). Returns the observations dict.
"""
function observe_and_update!(
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    true_terrain::Matrix{Float64},
    initial_mean_grid::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    rng::AbstractRNG;
    update_mode::Symbol = GLOBAL_UPDATE_MODE,
    decay_lambda::Float64 = GLOBAL_DECAY_LAMBDA,
)
    if update_mode == :deterministic
        # Deterministic observation: no noise, exact up to weight w
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
                    update_val = true_terrain[i, j] - initial_mean_grid[i, j]
                    mean_grid[i, j] = initial_mean_grid[i, j] + w * update_val
                end
            end
        end

        if altitude == 0
            grid_std[i_curr, j_curr] = 0.0
            mean_grid[i_curr, j_curr] = true_terrain[i_curr, j_curr]
        end

        return nothing
    elseif update_mode == :wholesale
        # Stochastic wholesale update (matches batch_run RNG consumption)
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
                    update_val = true_terrain[i, j] - initial_mean_grid[i, j]
                    alt_obsv_mean = initial_mean_grid[i, j] + w * update_val
                    mean_grid[i, j] = alt_obsv_mean + new_std * randn(rng)
                end
            end
        end

        if altitude == 0
            grid_std[i_curr, j_curr] = 0.0
            mean_grid[i_curr, j_curr] = true_terrain[i_curr, j_curr]
        end

        return nothing
    else
        # Bayesian modes: generate all observations then update
        observations, obs_std = generate_cone_observation(
            true_terrain, initial_mean_grid, current_pos, altitude, noise_sigma,
            cone_angle, z_update, transition_k, rng)

        w = update_weight(altitude, z_update, transition_k)
        bayesian_update!(mean_grid, grid_std, observations, obs_std;
                         mode=update_mode, lambda=decay_lambda, obs_weight=w)

        return observations
    end
end

# ─────────────────────────────────────────────────────────────────────
#  LEGACY WRAPPERS (for backward compatibility)
# ─────────────────────────────────────────────────────────────────────

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
    true_terrain = initial_mean_grid .+ update_grid
    observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid, current_pos,
                        altitude, noise_sigma, cone_angle, z_update, transition_k, rng)
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
        for jj in max(1, j-di):min(ncols, j+di)
            if abs(ii-i) + abs(jj-j) <= z && mean_grid[ii, jj] > best_val
                best_val  = mean_grid[ii, jj]
                best_cell = (ii, jj)
            end
        end
    end
    return greedy_step_toward((i, j), best_cell)
end

end  # _ENV_LOADED guard
