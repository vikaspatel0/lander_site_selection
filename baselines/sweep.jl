# =====================================================================
#  Full parameter sweep: policy × k × σ
#
#  Produces a results matrix that can be visualized as heatmaps,
#  matching the format of image.png / toy_runs_analysis_fast.m
#
#  Outputs CSV files to data_large/ for MATLAB compatibility,
#  plus a consolidated text summary.
# =====================================================================

include(joinpath(@__DIR__, "..", "environment.jl"))
include("ucb.jl")
include("lcb.jl")
include("lucb.jl")
include("thompson.jl")
include("greedy_risk.jl")
include("mcts_rollout.jl")
include("mcts_tree.jl")
include("tail_lookahead.jl")

using Printf
using DelimitedFiles

# ─────────────────────────────────────────────────────────────────────
#  Metrics (from toy_runs_analysis_fast.m)
# ─────────────────────────────────────────────────────────────────────

function compute_metrics(vals::Vector{Float64}; beta::Float64=0.5)
    m    = mean(vals)
    ent  = -(1.0 / beta) * log(mean(exp.(-beta .* vals)) + eps())
    sv   = sort(vals)
    n10  = max(1, ceil(Int, 0.10 * length(vals)))
    cvar = mean(sv[1:n10])
    q05  = length(vals) >= 20 ? quantile(vals, 0.05) : minimum(vals)
    mn   = minimum(vals)
    return (mean=m, entropic=ent, cvar10=cvar, q05=q05, min=mn)
end

# ─────────────────────────────────────────────────────────────────────
#  Build policy list (same as run_all.jl but parameterized)
# ─────────────────────────────────────────────────────────────────────

function build_sweep_policies(; z_update::Int, transition_k::Float64,
                                noise_sigma::Float64, cone_angle::Float64,
                                n_rollouts::Int, budget::ActionBudget)
    policies = Tuple{String, Function}[]

    function gr(name, rc; λe=0.0)
        push!(policies, (name, (img, ug, ns, ss, ca) ->
            plan_greedy_risk(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=rc, λ_explore=λe,
                ts_rng=MersenneTwister(rand(1:10^9)))))
    end

    # --- Greedy-risk: Entropic-sigma variants ---
    gr("Max σ 0.5",        RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMax))
    gr("Min σ 0.5",        RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMin))

    # --- Greedy-risk: CVaR ---
    gr("CVaR α 10",        RiskConfig(mode=RiskCVaR, alpha=0.10))
    gr("CVaR α 30",        RiskConfig(mode=RiskCVaR, alpha=0.30))
    gr("CVaR α 50",        RiskConfig(mode=RiskCVaR, alpha=0.50))

    # --- Greedy-risk: with exploration bonus ---
    gr("Max σ 0 Expl",     RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMax); λe=0.1)
    gr("Min σ 0 Expl",     RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMin); λe=0.1)
    gr("CVaR α 10 Expl",   RiskConfig(mode=RiskCVaR, alpha=0.10); λe=0.1)
    gr("CVaR α 30 Expl",   RiskConfig(mode=RiskCVaR, alpha=0.30); λe=0.1)
    gr("CVaR α 50 Expl",   RiskConfig(mode=RiskCVaR, alpha=0.50); λe=0.1)

    # --- Greedy-risk: EVaR (full sweep) ---
    gr("EVaR α 0.01",      RiskConfig(mode=RiskEVaR, alpha=0.01); λe=0.1)
    gr("EVaR α 0.05",      RiskConfig(mode=RiskEVaR, alpha=0.05); λe=0.1)
    gr("EVaR α 0.1",       RiskConfig(mode=RiskEVaR, alpha=0.10); λe=0.1)
    gr("EVaR α 0.2",       RiskConfig(mode=RiskEVaR, alpha=0.20); λe=0.1)
    gr("EVaR α 0.3",       RiskConfig(mode=RiskEVaR, alpha=0.30); λe=0.1)
    gr("EVaR α 0.4",       RiskConfig(mode=RiskEVaR, alpha=0.40); λe=0.1)
    gr("EVaR α 0.5",       RiskConfig(mode=RiskEVaR, alpha=0.50); λe=0.1)

    # --- Greedy-risk: Cellwise sigma ---
    gr("C-wise σ 0",       RiskConfig(mode=RiskEntropicSigma, beta=0.5, use_cellwise_sigma=true))

    # --- Tail lookahead: %ile (ConstP) × lambda_best ---
    for λb in [0.5, 1.0, 1.5]
        λb_str = @sprintf("%.1f", λb)
        push!(policies, ("TL %ile λb=$λb_str", (img, ug, ns, ss, ca) ->
            plan_tail_lookahead(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=TLRiskConfig(mode=TLRiskConstP, p_const=0.9),
                tail_cfg=TailLookaheadConfig(tail_fraction=0.10, lambda_tail=0.025,
                    lambda_best=λb, lambda_travel=0.01, lambda_entropy=0.1,
                    simulate_next_observation=true))))
    end

    # --- Tail lookahead: mean × lambda_best ---
    for λb in [0.5, 1.0, 1.5]
        λb_str = @sprintf("%.1f", λb)
        push!(policies, ("TL mean λb=$λb_str", (img, ug, ns, ss, ca) ->
            plan_tail_lookahead(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=TLRiskConfig(mode=TLRiskMean),
                tail_cfg=TailLookaheadConfig(tail_fraction=0.10, lambda_tail=0.025,
                    lambda_best=λb, lambda_travel=0.01, lambda_entropy=0.1,
                    simulate_next_observation=true))))
    end

    # --- Tail lookahead: 80% tail × lambda_best ---
    for λb in [0.5, 1.0, 1.5]
        λb_str = @sprintf("%.1f", λb)
        push!(policies, ("TL 80% λb=$λb_str", (img, ug, ns, ss, ca) ->
            plan_tail_lookahead(img, ug, ns, ss, ca;
                z_update=z_update, transition_k=transition_k,
                risk_cfg=TLRiskConfig(mode=TLRiskEntropicSigma, beta=0.5),
                tail_cfg=TailLookaheadConfig(tail_fraction=0.80, lambda_tail=0.025,
                    lambda_best=λb, lambda_travel=0.01, lambda_entropy=0.1,
                    simulate_next_observation=true))))
    end

    # Bandit
    push!(policies, ("UCB α=0.5", (img, ug, ns, ss, ca) ->
        plan_ucb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=0.5)))
    push!(policies, ("UCB α=1.0", (img, ug, ns, ss, ca) ->
        plan_ucb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=1.0)))
    push!(policies, ("LCB α=1.0", (img, ug, ns, ss, ca) ->
        plan_lcb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=1.0)))
    push!(policies, ("LUCB", (img, ug, ns, ss, ca) ->
        plan_lucb(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k, α=1.0, c=2.0)))
    push!(policies, ("Thompson (ban)", (img, ug, ns, ss, ca) ->
        plan_thompson(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                      ts_rng=MersenneTwister(rand(1:10^9)))))

    # MCTS rollout
    push!(policies, ("MCTS-ro greedy", (img, ug, ns, ss, ca) ->
        plan_mcts_rollout(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                          n_rollouts=n_rollouts, rollout_policy=:greedy,
                          sample_rng=MersenneTwister(rand(1:10^9)), budget=budget)))
    push!(policies, ("MCTS-ro ucb", (img, ug, ns, ss, ca) ->
        plan_mcts_rollout(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                          n_rollouts=n_rollouts, rollout_policy=:ucb, rollout_alpha=1.0,
                          sample_rng=MersenneTwister(rand(1:10^9)), budget=budget)))

    # MCTS tree (greedy + ucb rollouts only — random is too weak)
    push!(policies, ("MCTS-tree greedy", (img, ug, ns, ss, ca) ->
        plan_mcts_tree(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                       mcts_cfg=MCTSTreeConfig(iterations=1500, rollout_policy=:greedy),
                       rng=MersenneTwister(rand(1:10^9)), budget=budget)))
    push!(policies, ("MCTS-tree ucb", (img, ug, ns, ss, ca) ->
        plan_mcts_tree(img, ug, ns, ss, ca; z_update=z_update, transition_k=transition_k,
                       mcts_cfg=MCTSTreeConfig(iterations=1500, rollout_policy=:ucb, rollout_alpha=1.0),
                       rng=MersenneTwister(rand(1:10^9)), budget=budget)))

    return policies
end

# ─────────────────────────────────────────────────────────────────────
#  Sweep runner
# ─────────────────────────────────────────────────────────────────────

function run_sweep(;
    n_sims::Int = 10,
    seed::Int = 1234,
    grid_size::Int = 60,
    start_state::Tuple{Int,Int,Int} = (20, 36, 30),
    cone_angle::Float64 = π/8,
    terrain_value_min::Float64 = 0.0,
    terrain_value_range::Float64 = 10.0,
    z_update::Int = 24,
    n_rollouts::Int = 200,
    budget::ActionBudget = DEFAULT_BUDGET,
    k_values::Vector{Float64} = [0.0, 2.0, 4.0, 7.0, 15.0],
    sigma_values::Vector{Float64} = [0.5, 1.0, 2.0, 3.0, 5.0],
    metric_name::String = "Entropic",  # which metric for the heatmap
)
    t0_total = time()

    # Get policy names from a dummy build
    dummy = build_sweep_policies(z_update=z_update, transition_k=0.0,
                                  noise_sigma=1.0, cone_angle=cone_angle,
                                  n_rollouts=n_rollouts, budget=budget)
    policy_names = [name for (name, _) in dummy]
    n_policies = length(policy_names)

    # Results: policy × k × σ
    heatmaps = Dict{String, Matrix{Float64}}()  # metric_name => (n_policies × n_k) per σ

    for (si, noise_sigma) in enumerate(sigma_values)
        scores = zeros(n_policies, length(k_values))

        for (ki, transition_k) in enumerate(k_values)
            @printf("\n── σ=%.1f, k=%.1f ──\n", noise_sigma, transition_k)

            policies = build_sweep_policies(
                z_update=z_update, transition_k=transition_k,
                noise_sigma=noise_sigma, cone_angle=cone_angle,
                n_rollouts=n_rollouts, budget=budget)

            results = [Float64[] for _ in 1:n_policies]
            rng = MersenneTwister(seed)

            t0 = time()
            for sim in 1:n_sims
                terrain_seed = rand(rng, 1:10^9)
                noise_seed   = rand(rng, 1:10^9)

                initial_mean_grid = generate_terrain_2(grid_size;
                    seed=terrain_seed, value_min=terrain_value_min,
                    value_range=terrain_value_range)

                rng_noise = MersenneTwister(noise_seed)
                update_grid = noise_sigma .* randn(rng_noise, grid_size, grid_size)

                for (pi, (_, plan_fn)) in enumerate(policies)
                    _, landing_value, _, _ = plan_fn(
                        initial_mean_grid, update_grid, noise_sigma,
                        start_state, cone_angle)
                    push!(results[pi], landing_value)
                end
            end
            elapsed = time() - t0

            for pi in 1:n_policies
                m = compute_metrics(results[pi])
                scores[pi, ki] = if metric_name == "Mean"
                    m.mean
                elseif metric_name == "Entropic"
                    m.entropic
                elseif metric_name == "CVaR10"
                    m.cvar10
                elseif metric_name == "Min"
                    m.min
                else
                    m.mean
                end
            end
            @printf("  done in %.1fs\n", elapsed)
        end

        heatmaps["σ=$(noise_sigma)"] = scores
    end

    # ── Print text heatmap ──
    k_labels = [@sprintf("k=%.0f", k) for k in k_values]

    println("\n", "="^100)
    @printf("  Sweep Results — Metric: %s, %d sims, grid=%d, z_update=%d\n",
            metric_name, n_sims, grid_size, z_update)
    println("="^100)

    for (si, noise_sigma) in enumerate(sigma_values)
        scores = heatmaps["σ=$(noise_sigma)"]
        @printf("\n  ── σ = %.1f ──\n", noise_sigma)
        @printf("  %-20s", "Policy")
        for kl in k_labels
            @printf(" %8s", kl)
        end
        println()
        println("  ", "─"^(20 + 9*length(k_values)))

        for pi in 1:n_policies
            @printf("  %-20s", policy_names[pi])
            for ki in 1:length(k_values)
                @printf(" %8.2f", scores[pi, ki])
            end
            println()
        end
    end

    elapsed_total = time() - t0_total
    @printf("\n  [Total sweep time: %.1fm]\n", elapsed_total / 60)
end

# ─────────────────────────────────────────────────────────────────────
#  Run sweep
# ─────────────────────────────────────────────────────────────────────

run_sweep(
    n_sims = 10,
    metric_name = "Entropic",
    budget = ActionBudget(time_limit_s=30.0, memory_limit_bytes=256*1024*1024),
    sigma_values = [5.0],
    k_values = [0.0, 2.0, 4.0, 7.0, 15.0],
)
