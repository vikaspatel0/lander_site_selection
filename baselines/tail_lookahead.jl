# =====================================================================
#  Tail Lookahead Planner
#
#  At each step, evaluates all 5 actions by simulating the next
#  observation and computing a combined score:
#    score = λ_tail · tail_mean + λ_best · best_score + λ_entropy · ΔH
#
#  tail_mean = mean of worst tail_fraction of reachable cell scores
#  Optimizes for robust "floor" of landing options.
# =====================================================================

# Requires: environment.jl loaded first

using Distributions: pdf, cdf, quantile, Normal

# ─────────────────────────────────────────────────────────────────────
#  Risk configuration (shared with greedy_risk.jl but self-contained)
# ─────────────────────────────────────────────────────────────────────

@enum TLRiskMode begin
    TLRiskConstP
    TLRiskEntropicSigma
    TLRiskCVaR
    TLRiskMean
end

Base.@kwdef struct TLRiskConfig
    mode::TLRiskMode = TLRiskConstP
    p_const::Float64 = 0.90
    beta::Float64 = 0.0
    alpha::Float64 = 0.1
end

function tl_risk_value(rc::TLRiskConfig, cell_mean::Float64, cell_std::Float64)
    if rc.mode == TLRiskCVaR
        if cell_std ≈ 0.0; return cell_mean; end
        q = quantile(Normal(0,1), rc.alpha)
        phi = pdf(Normal(0,1), q)
        return cell_mean - cell_std * (phi / rc.alpha)
    elseif rc.mode == TLRiskConstP
        if cell_std ≈ 0.0; return cell_mean; end
        return quantile(Normal(cell_mean, cell_std), rc.p_const)
    elseif rc.mode == TLRiskEntropicSigma
        p = cdf(Normal(0,1), -(rc.beta / 2.0) * cell_std)
        if cell_std ≈ 0.0; return cell_mean; end
        return quantile(Normal(cell_mean, cell_std), p)
    elseif rc.mode == TLRiskMean
        return cell_mean
    else
        error("Unknown TLRiskMode")
    end
end

function gaussian_entropy(std::Float64)
    σ = max(std, 1e-6)
    return 0.5 * log(2π * exp(1) * σ^2)
end

# ─────────────────────────────────────────────────────────────────────
#  Tail lookahead configuration
# ─────────────────────────────────────────────────────────────────────

Base.@kwdef struct TailLookaheadConfig
    tail_fraction::Float64 = 0.10
    lambda_tail::Float64 = 1.0
    lambda_best::Float64 = 0.25
    lambda_travel::Float64 = 0.01
    lambda_entropy::Float64 = 0.0
    simulate_next_observation::Bool = true
end

# ─────────────────────────────────────────────────────────────────────
#  Core tail metrics computation
# ─────────────────────────────────────────────────────────────────────

function collective_tail_metrics(
    mean_grid::Matrix{Float64},
    std_grid::Matrix{Float64},
    pos::Tuple{Int,Int},
    z::Int,
    risk_cfg::TLRiskConfig,
    cfg::TailLookaheadConfig
)
    nrows, ncols = size(mean_grid)
    candidates = reachable_indices(nrows, ncols, pos[1], pos[2], z)

    scores = Float64[]
    best_score = -Inf
    best_cell = pos
    entropy_sum = 0.0

    for (ti, tj) in candidates
        cell_score = tl_risk_value(risk_cfg, mean_grid[ti, tj], std_grid[ti, tj])
        cell_score -= cfg.lambda_travel * manhattan(pos, (ti, tj))
        push!(scores, cell_score)

        if cell_score > best_score
            best_score = cell_score
            best_cell = (ti, tj)
        end
        entropy_sum += gaussian_entropy(std_grid[ti, tj])
    end

    sort!(scores)
    m = max(1, ceil(Int, cfg.tail_fraction * length(scores)))
    tail_mean = mean(@view scores[1:m])

    combined = cfg.lambda_tail * tail_mean + cfg.lambda_best * best_score
    return combined, best_cell, tail_mean, best_score, entropy_sum
end

# ─────────────────────────────────────────────────────────────────────
#  Bayesian expected std reduction for lookahead simulation
#  Given prior std σ_prior and obs_std, posterior std is:
#    σ_post = 1/sqrt(1/σ_prior² + 1/obs_std²)
# ─────────────────────────────────────────────────────────────────────

function bayesian_expected_std_update!(
    std_grid::Matrix{Float64},
    current_pos::Tuple{Int,Int},
    altitude::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64
)
    nrows, ncols = size(std_grid)
    w = update_weight(altitude, z_update, transition_k)
    obs_std = max(noise_sigma * (1.0 - w), 0.0)

    if altitude == 0
        i, j = current_pos
        std_grid[i, j] = 0.0
        return nothing
    end

    cells = cone_cells(current_pos, altitude, cone_angle, nrows, ncols)
    for (i, j) in cells
        σ_prior = std_grid[i, j]
        if σ_prior <= 0.0
            continue
        end
        if obs_std <= 0.0
            std_grid[i, j] = 0.0
        else
            τ_prior = 1.0 / (σ_prior^2)
            τ_obs = 1.0 / (obs_std^2)
            std_grid[i, j] = 1.0 / sqrt(τ_prior + τ_obs)
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
#  Lookahead simulation: sample terrain from belief, observe, update
# ─────────────────────────────────────────────────────────────────────

function simulate_lookahead_observation!(
    mean_next::Matrix{Float64},
    std_next::Matrix{Float64},
    pos::Tuple{Int,Int},
    alt::Int,
    noise_sigma::Float64,
    cone_angle::Float64,
    z_update::Int,
    transition_k::Float64,
    sim_rng::AbstractRNG
)
    nrows, ncols = size(mean_next)
    # Sample terrain from current belief
    sampled_terrain = Matrix{Float64}(undef, nrows, ncols)
    for i in 1:nrows, j in 1:ncols
        sampled_terrain[i, j] = mean_next[i, j] + std_next[i, j] * randn(sim_rng)
    end
    # Generate observations from the sampled terrain and update belief
    observations, obs_std = generate_cone_observation(
        sampled_terrain, mean_next, pos, alt, noise_sigma, cone_angle, z_update, transition_k, sim_rng)
    bayesian_update!(mean_next, std_next, observations, obs_std)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
#  Tail lookahead planner
# ─────────────────────────────────────────────────────────────────────

function plan_tail_lookahead(
    initial_mean_grid::Matrix{Float64},
    update_grid::Matrix{Float64},
    noise_sigma::Float64,
    start_state::Tuple{Int,Int,Int},
    cone_angle::Float64;
    z_update::Int = start_state[3],
    transition_k::Float64 = 0.0,
    risk_cfg::TLRiskConfig = TLRiskConfig(mode=TLRiskConstP, p_const=0.9),
    tail_cfg::TailLookaheadConfig = TailLookaheadConfig(),
    obs_rng::AbstractRNG = MersenneTwister(0),
)
    nrows, ncols = size(initial_mean_grid)
    true_terrain = initial_mean_grid .+ update_grid
    mean_grid = copy(initial_mean_grid)
    grid_std  = fill(noise_sigma, nrows, ncols)

    # Separate RNG for lookahead simulations so it doesn't interfere with obs_rng
    sim_rng = MersenneTwister(hash(obs_rng))

    trajectory = TrajectoryStep[]
    current_state = start_state

    while current_state[3] > 0
        i, j, z = current_state

        observe_and_update!(mean_grid, grid_std, true_terrain, initial_mean_grid,
                            (i, j), z, noise_sigma, cone_angle, z_update, transition_k, obs_rng)

        # Evaluate each action by lookahead
        best_action = :none
        best_target = (i, j)
        best_total_score = -Inf

        for a in ACTIONS
            sp = step_next_state(nrows, ncols, current_state, a)
            ip, jp, zp = sp

            reachable_sp = reachable_indices(nrows, ncols, ip, jp, zp)
            entropy_before = sum(gaussian_entropy(grid_std[u, v]) for (u, v) in reachable_sp)

            mean_next = copy(mean_grid)
            std_next  = copy(grid_std)

            if tail_cfg.simulate_next_observation
                # Simulate observation using sampled terrain from belief
                simulate_lookahead_observation!(
                    mean_next, std_next, (ip, jp), zp,
                    noise_sigma, cone_angle, z_update, transition_k, sim_rng)
            end

            coll_score, target, _, _, entropy_after = collective_tail_metrics(
                mean_next, std_next, (ip, jp), zp, risk_cfg, tail_cfg)

            entropy_reduction = entropy_before - entropy_after
            ap = action_penalty(a)
            total_score = coll_score + tail_cfg.lambda_entropy * entropy_reduction + ap

            if total_score > best_total_score
                best_total_score = total_score
                best_action = a
                best_target = target
            end
        end

        next_state = step_next_state(nrows, ncols, current_state, best_action)

        r = if next_state[3] == 0
            true_val = true_terrain[next_state[1], next_state[2]]
            true_val + action_penalty(best_action)
        else
            action_penalty(best_action)
        end

        push!(trajectory, (state=current_state, action=best_action, reward=r,
                           next_state=next_state,
                           grid_std_snapshot=copy(grid_std),
                           grid_mean_snapshot=copy(mean_grid),
                           target=best_target))

        current_state = next_state
    end

    land_i, land_j, _ = trajectory[end].next_state
    landing_value = true_terrain[land_i, land_j]
    return trajectory, landing_value, grid_std, mean_grid
end
