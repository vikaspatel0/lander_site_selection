# =====================================================================
#  Greedy Target with Risk-Sensitive Reward
#
#  Unified baseline covering all risk configurations. The planner is
#  always: score each reachable cell, pick best, move one step toward it.
#
#  Risk modes:
#    RiskConstP         — quantile(Normal(μ,σ), p)
#    RiskEntropicSigma  — adaptive percentile based on surrounding σ
#    RiskCVaR           — conditional value at risk
#    RiskEVaR           — entropic value at risk (tighter bound)
#    RiskThompson       — sample from N(μ,σ), pick best sample
#
#  Optional exploration bonus (λ_explore):
#    score += λ_explore · expected_σ_reduction_for_action
#
#  Requires: environment.jl loaded first
# =====================================================================

using Distributions: pdf, cdf, quantile, Normal

# ─────────────────────────────────────────────────────────────────────
#  Risk configuration
# ─────────────────────────────────────────────────────────────────────

@enum RiskMode begin
    RiskConstP
    RiskEntropicSigma
    RiskCVaR
    RiskEVaR
    RiskThompson
end

@enum SigmaRefMode begin
    SigmaMax
    SigmaMean
    SigmaMin
end

Base.@kwdef struct RiskConfig
    mode::RiskMode = RiskConstP
    p_const::Float64 = 0.90           # RiskConstP: fixed percentile
    beta::Float64 = 0.5               # RiskEntropicSigma: entropic strength
    sigma_ref_mode::SigmaRefMode = SigmaMean  # RiskEntropicSigma: how to compute σ_ref
    use_cellwise_sigma::Bool = false   # RiskEntropicSigma: use cell's own σ instead of reachable σ_ref
    alpha::Float64 = 0.1              # RiskCVaR / RiskEVaR: tail fraction
end

# ─────────────────────────────────────────────────────────────────────
#  Risk value functions
# ─────────────────────────────────────────────────────────────────────

function entropic_percentile(beta::Float64, sigma_ref::Float64)
    cdf(Normal(0,1), -(beta/2.0) * sigma_ref)
end

function risk_sensitive_reward(mu::Float64, std::Float64, percentile::Float64)
    if std ≈ 0.0
        return mu
    end
    quantile(Normal(mu, std), percentile)
end

function cvar_normal(mu::Float64, std::Float64, alpha::Float64)
    if std ≈ 0.0
        return mu
    end
    q = quantile(Normal(0, 1), alpha)
    phi = pdf(Normal(0, 1), q)
    return mu - std * (phi / alpha)
end

function evar_normal(mu::Float64, std::Float64, alpha::Float64)
    if std ≈ 0.0
        return mu
    end
    k = sqrt(2 * log(1/alpha))
    return mu - std * k
end

function select_sigma_ref(grid_std::Matrix{Float64},
                          reachable::Vector{Tuple{Int,Int}},
                          mode::SigmaRefMode)
    sigmas = [grid_std[ii, jj] for (ii, jj) in reachable]
    if mode == SigmaMax
        return maximum(sigmas)
    elseif mode == SigmaMean
        return mean(sigmas)
    else
        return minimum(sigmas)
    end
end

function compute_risk_value(rc::RiskConfig,
                            cell_mean::Float64,
                            cell_std::Float64;
                            grid_std::Union{Nothing, Matrix{Float64}} = nothing,
                            reachable::Union{Nothing, Vector{Tuple{Int,Int}}} = nothing,
                            ts_rng::Union{Nothing, AbstractRNG} = nothing)
    if rc.mode == RiskCVaR
        return cvar_normal(cell_mean, cell_std, rc.alpha)
    elseif rc.mode == RiskEVaR
        return evar_normal(cell_mean, cell_std, rc.alpha)
    elseif rc.mode == RiskThompson
        return cell_mean + cell_std * randn(ts_rng)
    elseif rc.mode == RiskConstP
        return risk_sensitive_reward(cell_mean, cell_std, rc.p_const)
    elseif rc.mode == RiskEntropicSigma
        if rc.use_cellwise_sigma
            p = entropic_percentile(rc.beta, cell_std)
        else
            sigma_ref = select_sigma_ref(grid_std, reachable, rc.sigma_ref_mode)
            p = entropic_percentile(rc.beta, sigma_ref)
        end
        return risk_sensitive_reward(cell_mean, cell_std, p)
    else
        error("Unknown RiskMode: $(rc.mode)")
    end
end

# ─────────────────────────────────────────────────────────────────────
#  Exploration bonus: expected σ reduction for an action
#  Under the Bayesian model, the posterior std for a cell with prior σ_prior
#  observed with obs_std is 1/sqrt(1/σ_prior² + 1/obs_std²).
#  The gain is the expected reduction in mean-weighted std.
# ─────────────────────────────────────────────────────────────────────

function expected_sigma_mean_gain(
    mean_grid::Matrix{Float64},
    grid_std::Matrix{Float64},
    sensed_pos::Tuple{Int,Int},
    sensed_alt::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64
)
    # Exploration gain: sum of (σ_prior - σ_post) * mean over cone cells
    # where σ_post < σ_prior
    nrows, ncols = size(grid_std)
    w = update_weight(sensed_alt, z_update, transition_k)
    new_std = max(noise_sigma * (1.0 - w), 0.0)

    cells = cone_cells(sensed_pos, sensed_alt, cone_angle, nrows, ncols)
    gain = 0.0
    for (i, j) in cells
        σ_prior = grid_std[i, j]
        if σ_prior <= 0.0 || new_std >= σ_prior
            continue
        end
        gain += (σ_prior - new_std) * mean_grid[i, j]
    end
    return gain
end

# ─────────────────────────────────────────────────────────────────────
#  Greedy target planner
# ─────────────────────────────────────────────────────────────────────

function plan_greedy_risk(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    risk_cfg::RiskConfig = RiskConfig(mode=RiskConstP, p_const=0.9),
    λ_travel::Float64 = 0.01,
    λ_explore::Float64 = 0.0,
    ts_rng::AbstractRNG = MersenneTwister(42),
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

        # Precompute exploration gains per action (if λ_explore > 0)
        action_gains = Dict{Symbol, Float64}()
        if λ_explore > 0.0
            for a in ACTIONS
                sp = step_next_state(nrows, ncols, current_state, a)
                action_gains[a] = expected_sigma_mean_gain(
                    mean_grid, grid_std, (sp[1], sp[2]), sp[3],
                    noise_sigma, cone_angle, z_update, transition_k)
            end
        end

        best_cell = (i, j)
        best_score = -Inf
        for (ri, rj) in reachable
            v = compute_risk_value(risk_cfg, mean_grid[ri, rj], grid_std[ri, rj];
                                   grid_std=grid_std, reachable=reachable, ts_rng=ts_rng)
            d = manhattan((i, j), (ri, rj))
            score = v - λ_travel * d
            if λ_explore > 0.0
                a = greedy_step_toward((i, j), (ri, rj))
                score += λ_explore * action_gains[a]
            end
            if score > best_score
                best_score = score
                best_cell  = (ri, rj)
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
