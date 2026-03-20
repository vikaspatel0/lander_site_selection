# =====================================================================
#  Unified baseline comparison runner
#
#  Runs all baselines on the same terrain seeds and reports:
#    Mean, SE, Entropic, CVaR10, Q05, Min, %Opt
#  with paired t-tests vs a chosen reference policy.
#
#  Includes greedy-risk, bandit, MCTS, and tail-lookahead baselines.
# =====================================================================

# Include shared environment from lander_site_selection root, then all baselines
include(joinpath(@__DIR__, "..", "environment.jl"))
include("ucb.jl")            # plan_ucb
include("lcb.jl")            # plan_lcb
include("lucb.jl")           # plan_lucb
include("thompson.jl")       # plan_thompson
include("greedy_risk.jl")    # plan_greedy_risk, RiskConfig, etc.
include("mcts_rollout.jl")   # plan_mcts_rollout
include("mcts_tree.jl")      # plan_mcts_tree, MCTSTreeConfig
include("tail_lookahead.jl") # plan_tail_lookahead, TailLookaheadConfig


using Printf

# ─────────────────────────────────────────────────────────────────────
#  Policy registry
#
#  Each policy lambda now takes 6 args: (img, ug, ns, ss, ca, obs_rng)
#  where obs_rng is the RNG for stochastic sensor observations.
# ─────────────────────────────────────────────────────────────────────

function build_policy_list(;
    z_update::Int,
    transition_k::Float64,
    noise_sigma::Float64,
    cone_angle::Float64,
    n_rollouts::Int = 200,
    budget::ActionBudget = DEFAULT_BUDGET,
    planning_seed::Union{Nothing,Int} = nothing,
)
    policies = Tuple{String, Function}[]
    planner_seed = isnothing(planning_seed) ? rand(MersenneTwister(1), 1:10^9) : planning_seed

    # Helper for greedy-risk policies (most common pattern)
    function gr(name, rc; λe=0.0)
        push!(policies, (name, (img, ug, ns, ss, ca, obs_rng) ->
            plan_greedy_risk(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=rc, λ_explore=λe,
                ts_rng=MersenneTwister(planner_seed),
                obs_rng=obs_rng)))
    end

    # ═══════════════════════════════════════════════════════════════════
    #  GROUP 1: Greedy-risk baselines
    # ═══════════════════════════════════════════════════════════════════

    # --- Entropic-sigma with different σ_ref modes ---
    gr("Max σ (β=0.5)",   RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMax))
    gr("Min σ (β=0.5)",   RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMin))
    gr("Mean σ (β=0.5)",  RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMean))

    # --- CVaR with different tail fractions ---
    gr("CVaR α=0.10",     RiskConfig(mode=RiskCVaR, alpha=0.10))
    gr("CVaR α=0.30",     RiskConfig(mode=RiskCVaR, alpha=0.30))
    gr("CVaR α=0.50",     RiskConfig(mode=RiskCVaR, alpha=0.50))

    # --- Entropic-sigma + exploration bonus ---
    gr("Max σ + Expl",    RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMax); λe=0.1)
    gr("Min σ + Expl",    RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMin); λe=0.1)

    # --- CVaR + exploration bonus ---
    gr("CVaR α=0.10 +Expl", RiskConfig(mode=RiskCVaR, alpha=0.10); λe=0.1)
    gr("CVaR α=0.30 +Expl", RiskConfig(mode=RiskCVaR, alpha=0.30); λe=0.1)
    gr("CVaR α=0.50 +Expl", RiskConfig(mode=RiskCVaR, alpha=0.50); λe=0.1)

    # --- EVaR ---
    gr("EVaR α=0.01",     RiskConfig(mode=RiskEVaR, alpha=0.01); λe=0.1)
    gr("EVaR α=0.10",     RiskConfig(mode=RiskEVaR, alpha=0.10); λe=0.1)
    gr("EVaR α=0.50",     RiskConfig(mode=RiskEVaR, alpha=0.50); λe=0.1)

    # --- Cellwise entropic ---
    gr("Entropic cellwise", RiskConfig(mode=RiskEntropicSigma, beta=0.5, use_cellwise_sigma=true))

    # --- Thompson (greedy-risk variant) ---
    gr("Thompson (greedy)", RiskConfig(mode=RiskThompson))

    # ═══════════════════════════════════════════════════════════════════
    #  GROUP 2: Tail lookahead
    # ═══════════════════════════════════════════════════════════════════

    for λb in [0.5, 1.0, 1.5]
        λb_str = @sprintf("%.1f", λb)
        push!(policies, ("TL λb=$λb_str", (img, ug, ns, ss, ca, obs_rng) ->
            plan_tail_lookahead(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=TLRiskConfig(mode=TLRiskEntropicSigma, beta=0.5),
                tail_cfg=TailLookaheadConfig(
                    tail_fraction=0.10,
                    lambda_tail=0.025,
                    lambda_best=λb,
                    lambda_travel=0.01,
                    lambda_entropy=0.1,
                    simulate_next_observation=true),
                obs_rng=obs_rng)))
    end

    # TL with different tail fractions
    for tf in [0.50, 0.80]
        tf_str = @sprintf("%.0f%%", 100*tf)
        push!(policies, ("TL $tf_str", (img, ug, ns, ss, ca, obs_rng) ->
            plan_tail_lookahead(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=TLRiskConfig(mode=TLRiskEntropicSigma, beta=0.5),
                tail_cfg=TailLookaheadConfig(
                    tail_fraction=tf,
                    lambda_tail=0.025,
                    lambda_best=1.0,
                    lambda_travel=0.01,
                    lambda_entropy=0.1,
                    simulate_next_observation=true),
                obs_rng=obs_rng)))
    end

    # ═══════════════════════════════════════════════════════════════════
    #  GROUP 3: Bandit-style target selection (our new baselines)
    # ═══════════════════════════════════════════════════════════════════

    push!(policies, ("BayesUCB α=0.5", (img, ug, ns, ss, ca, obs_rng) ->
        plan_ucb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=0.5,
                 obs_rng=obs_rng)))
    push!(policies, ("BayesUCB α=1.0", (img, ug, ns, ss, ca, obs_rng) ->
        plan_ucb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=1.0,
                 obs_rng=obs_rng)))
    push!(policies, ("BayesLCB α=1.0", (img, ug, ns, ss, ca, obs_rng) ->
        plan_lcb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=1.0,
                 obs_rng=obs_rng)))
    push!(policies, ("BayesLUCB α=1 c=2", (img, ug, ns, ss, ca, obs_rng) ->
        plan_lucb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=1.0, c=2.0,
                  obs_rng=obs_rng)))
    push!(policies, ("Thompson (bandit)", (img, ug, ns, ss, ca, obs_rng) ->
        plan_thompson(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                      ts_rng=MersenneTwister(planner_seed),
                      obs_rng=obs_rng)))

    # ═══════════════════════════════════════════════════════════════════
    #  GROUP 4: MCTS planners (with time/memory budget)
    # ═══════════════════════════════════════════════════════════════════

    push!(policies, ("MCTS-rollout greedy", (img, ug, ns, ss, ca, obs_rng) ->
        plan_mcts_rollout(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                          n_rollouts=n_rollouts, rollout_policy=:greedy,
                          sample_rng=MersenneTwister(planner_seed),
                          obs_rng=obs_rng, budget=budget)))
    push!(policies, ("MCTS-rollout ucb α=1", (img, ug, ns, ss, ca, obs_rng) ->
        plan_mcts_rollout(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                          n_rollouts=n_rollouts, rollout_policy=:ucb, rollout_alpha=1.0,
                          sample_rng=MersenneTwister(planner_seed),
                          obs_rng=obs_rng, budget=budget)))
    push!(policies, ("MCTS-rollout coneinfo", (img, ug, ns, ss, ca, obs_rng) ->
        plan_mcts_rollout(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                          n_rollouts=n_rollouts, rollout_policy=:coneinfo, rollout_alpha=0.5,
                          sample_rng=MersenneTwister(planner_seed),
                          obs_rng=obs_rng, budget=budget)))
    push!(policies, ("MCTS-tree greedy", (img, ug, ns, ss, ca, obs_rng) ->
        plan_mcts_tree(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                       mcts_cfg=MCTSTreeConfig(iterations=1500, exploration_c=1.4,
                                                max_rollout_steps=2000, rollout_policy=:greedy),
                       rng=MersenneTwister(planner_seed),
                       obs_rng=obs_rng, budget=budget)))
    push!(policies, ("MCTS-tree ucb", (img, ug, ns, ss, ca, obs_rng) ->
        plan_mcts_tree(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                       mcts_cfg=MCTSTreeConfig(iterations=1500, exploration_c=1.4,
                                                max_rollout_steps=2000, rollout_policy=:ucb,
                                                rollout_alpha=1.0),
                       rng=MersenneTwister(planner_seed),
                       obs_rng=obs_rng, budget=budget)))

    return policies
end

# ─────────────────────────────────────────────────────────────────────
#  Batch runner
# ─────────────────────────────────────────────────────────────────────

function run_comparison(;
    n_sims::Int = 10,
    seed::Int = 1234,
    grid_size::Int = 60,
    start_state::Tuple{Int,Int,Int} = (20, 36, 30),
    cone_angle::Float64 = π/8,
    noise_sigma::Float64 = 2.0,
    terrain_value_min::Float64 = 0.0,
    terrain_value_range::Float64 = 10.0,
    z_update::Int = 24,
    transition_k::Float64 = 0.0,
    n_rollouts::Int = 200,
    budget::ActionBudget = DEFAULT_BUDGET,
    reference_policy::String = "MCTS-rollout greedy",
    update_mode::Symbol = :altitude_weighted,
    decay_lambda::Float64 = 0.6,
)
    # Set global update mode so all observe_and_update! calls use it
    global GLOBAL_UPDATE_MODE = update_mode
    global GLOBAL_DECAY_LAMBDA = decay_lambda

    policies = build_policy_list(
        z_update=z_update, transition_k=transition_k,
        noise_sigma=noise_sigma, cone_angle=cone_angle,
        n_rollouts=n_rollouts, budget=budget)

    n_policies = length(policies)
    results = [Float64[] for _ in 1:n_policies]
    optimal_count = zeros(Int, n_policies)

    rng = MersenneTwister(seed)
    rng_planning = MersenneTwister(seed + 2000)

    t0 = time()
    for sim in 1:n_sims
        terrain_seed = rand(rng, 1:10^9)
        noise_seed   = rand(rng, 1:10^9)
        planning_seed = rand(rng_planning, 1:10^9)

        policies_for_sim = build_policy_list(
            z_update=z_update, transition_k=transition_k,
            noise_sigma=noise_sigma, cone_angle=cone_angle,
            n_rollouts=n_rollouts, budget=budget,
            planning_seed=planning_seed)

        initial_mean_grid = generate_terrain_2(grid_size;
            seed=terrain_seed, value_min=terrain_value_min, value_range=terrain_value_range)

        rng_noise = MersenneTwister(noise_seed)
        update_grid = noise_sigma .* randn(rng_noise, grid_size, grid_size)

        true_terrain = initial_mean_grid .+ update_grid
        optimal_value = maximum(true_terrain)

        for (pi, (name, plan_fn)) in enumerate(policies_for_sim)
            # Use planning_seed for obs_rng so all policies see the same observation noise
            obs_rng = MersenneTwister(planning_seed)

            _, landing_value, _, _ = plan_fn(
                initial_mean_grid, update_grid, noise_sigma,
                start_state, cone_angle, obs_rng)

            push!(results[pi], landing_value)
            if abs(landing_value - optimal_value) < 1e-10
                optimal_count[pi] += 1
            end
        end

        if sim % max(1, n_sims ÷ 4) == 0
            elapsed = time() - t0
            @printf("  sim %d/%d  (%.1fs elapsed)\n", sim, n_sims, elapsed)
        end
    end

    # ── Report ──
    beta = 0.5  # entropic risk parameter (matches toy_runs_analysis_fast.m)

    println("\n", "="^95)
    @printf("  Baseline Comparison: %d sims, grid=%d, σ=%.1f, k=%.1f, z_update=%d, budget=%.0fs/%.0fMB\n",
            n_sims, grid_size, noise_sigma, transition_k, z_update,
            budget.time_limit_s, budget.memory_limit_bytes / 1024 / 1024)
    println("="^95)

    @printf("\n  %-25s %10s %8s %8s %8s %8s %7s\n",
            "Policy", "Mean ± SE", "Entrop", "P05", "P10", "Min", "%Opt")
    println("  ", "─"^82)

    for (pi, (name, _)) in enumerate(policies)
        vals = results[pi]
        m = mean(vals)
        se = std(vals) / sqrt(length(vals))
        ent = -(1.0 / beta) * log(mean(exp.(-beta .* vals)) + eps())
        p05 = length(vals) >= 20 ? quantile(vals, 0.05) : minimum(vals)
        p10 = length(vals) >= 10 ? quantile(vals, 0.10) : minimum(vals)
        mn = minimum(vals)
        popt = 100.0 * optimal_count[pi] / n_sims

        @printf("  %-25s %6.2f±%4.2f %8.2f %8.2f %8.2f %8.2f %6.1f%%\n",
                name, m, se, ent, p05, p10, mn, popt)
    end

    # ── Paired t-tests vs reference ──
    ref_idx = findfirst(x -> x[1] == reference_policy, policies)
    if !isnothing(ref_idx)
        println("\n  Paired t-tests vs $(reference_policy):")
        println("  ", "─"^82)

        for (pi, (name, _)) in enumerate(policies)
            pi == ref_idx && continue
            diffs = results[pi] .- results[ref_idx]
            d_mean = mean(diffs)
            d_se   = std(diffs) / sqrt(length(diffs))
            t_stat = d_se > 0 ? d_mean / d_se : 0.0
            p_val = 2.0 * cdf(Normal(0,1), -abs(t_stat))
            sig = p_val < 0.001 ? "***" : p_val < 0.01 ? "**" : p_val < 0.05 ? "*" : ""

            @printf("  %-25s Δ=%6.3f ± %5.3f  t=%5.2f  p=%.4f  %s\n",
                    name, d_mean, d_se, t_stat, p_val, sig)
        end
    end

    elapsed = time() - t0
    @printf("\n  [Total time: %.1fm]\n", elapsed / 60)
end

# ─────────────────────────────────────────────────────────────────────
#  Default run
# ─────────────────────────────────────────────────────────────────────

run_comparison(
    n_sims = 20,
    seed = 1234,
    grid_size = 60,
    start_state = (20, 36, 30),
    cone_angle = π/8,
    noise_sigma = 2.0,
    terrain_value_min = 0.0,
    terrain_value_range = 10.0,
    z_update = 24,
    transition_k = 0.0,
    n_rollouts = 200,
    budget = ActionBudget(time_limit_s=30.0, memory_limit_bytes=256*1024*1024),
    update_mode = :altitude_weighted,
    decay_lambda = 0.6,
)
