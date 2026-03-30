using Printf, Statistics
include(joinpath(@__DIR__, "..", "environment.jl"))
include("ucb.jl"); include("lcb.jl"); include("lucb.jl"); include("thompson.jl")
include("greedy_risk.jl"); include("mcts_rollout.jl"); include("mcts_tree.jl")

function run_suite(; update_mode, label, n_sims=500, noise_sigma=5.0)
    global GLOBAL_UPDATE_MODE = update_mode
    global GLOBAL_DECAY_LAMBDA = 0.6
    seed = 1234; beta = 0.5

    rng = MersenneTwister(seed); rng_p = MersenneTwister(seed + 2000)
    tseeds = Int[]; nseeds = Int[]; pseeds = Int[]
    for _ in 1:n_sims
        push!(tseeds, rand(rng, 1:10^9))
        push!(nseeds, rand(rng, 1:10^9))
        push!(pseeds, rand(rng_p, 1:10^9))
    end

    policies = Tuple{String, Function}[]

    # Greedy-risk (no explore)
    for (name, rc) in [
        ("Max σ",       RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMax)),
        ("Min σ",       RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMin)),
        ("Mean σ",      RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMean)),
        ("C-wise σ",    RiskConfig(mode=RiskEntropicSigma, beta=0.5, use_cellwise_sigma=true)),
        ("CVaR α=0.10", RiskConfig(mode=RiskCVaR, alpha=0.10)),
        ("CVaR α=0.30", RiskConfig(mode=RiskCVaR, alpha=0.30)),
        ("CVaR α=0.50", RiskConfig(mode=RiskCVaR, alpha=0.50)),
        ("EVaR α=0.01", RiskConfig(mode=RiskEVaR, alpha=0.01)),
        ("EVaR α=0.05", RiskConfig(mode=RiskEVaR, alpha=0.05)),
        ("EVaR α=0.10", RiskConfig(mode=RiskEVaR, alpha=0.10)),
    ]
        push!(policies, (name, (img,ug,orng) -> plan_greedy_risk(img,ug,noise_sigma,(20,36,30),pi/8;
            z_update=24,transition_k=0.0, risk_cfg=rc, λ_explore=0.0, obs_rng=orng)))
    end

    # Greedy-risk (with explore)
    for (name, rc) in [
        ("Max σ +Expl",      RiskConfig(mode=RiskEntropicSigma, beta=0.5, sigma_ref_mode=SigmaMax)),
        ("CVaR α=0.10 +Expl", RiskConfig(mode=RiskCVaR, alpha=0.10)),
        ("EVaR α=0.05 +Expl", RiskConfig(mode=RiskEVaR, alpha=0.05)),
    ]
        push!(policies, (name, (img,ug,orng) -> plan_greedy_risk(img,ug,noise_sigma,(20,36,30),pi/8;
            z_update=24,transition_k=0.0, risk_cfg=rc, λ_explore=0.1, obs_rng=orng)))
    end

    # Bayesian bandit algorithms
    push!(policies, ("BayesUCB α=0.5", (img,ug,orng) -> plan_ucb(img,ug,noise_sigma,(20,36,30),pi/8; z_update=24,transition_k=0.0, α=0.5, obs_rng=orng)))
    push!(policies, ("BayesUCB α=1.0", (img,ug,orng) -> plan_ucb(img,ug,noise_sigma,(20,36,30),pi/8; z_update=24,transition_k=0.0, α=1.0, obs_rng=orng)))
    push!(policies, ("BayesLCB α=1.0", (img,ug,orng) -> plan_lcb(img,ug,noise_sigma,(20,36,30),pi/8; z_update=24,transition_k=0.0, α=1.0, obs_rng=orng)))
    push!(policies, ("BayesLUCB",      (img,ug,orng) -> plan_lucb(img,ug,noise_sigma,(20,36,30),pi/8; z_update=24,transition_k=0.0, α=1.0, c=2.0, obs_rng=orng)))
    push!(policies, ("Thompson",       (img,ug,orng) -> plan_thompson(img,ug,noise_sigma,(20,36,30),pi/8; z_update=24,transition_k=0.0, ts_rng=MersenneTwister(rand(orng,UInt64)), obs_rng=orng)))

    # MCTS (only 50 sims — too expensive for 500)
    n_mcts = min(n_sims, 50)

    println("\n", "="^80)
    @printf("  %s — σ=%.1f, k=0, %d sims (%d for MCTS)\n", label, noise_sigma, n_sims, n_mcts)
    println("="^80)
    @printf("  %-22s  %8s  %8s  %8s  %8s\n", "Policy", "Entrop", "P05", "P10", "Mean")
    println("  ", "-"^60)

    for (name, plan_fn) in policies
        is_mcts = startswith(name, "MCTS")
        ns = is_mcts ? n_mcts : n_sims
        vals = Float64[]
        for sim in 1:ns
            img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
            ug = noise_sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
            _, lv, _, _ = plan_fn(img, ug, MersenneTwister(pseeds[sim]))
            push!(vals, lv)
        end
        ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
        n_tag = is_mcts ? " ($(ns))" : ""
        @printf("  %-22s  %8.2f  %8.2f  %8.2f  %8.2f%s\n",
            name, ent, quantile(vals,0.05), quantile(vals,0.10), mean(vals), n_tag)
    end

    # Add MCTS separately
    mcts_policies = [
        ("MCTS-ro greedy", (img,ug,orng) -> plan_mcts_rollout(img,ug,noise_sigma,(20,36,30),pi/8;
            z_update=24,transition_k=0.0, n_rollouts=200, rollout_policy=:greedy,
            sample_rng=MersenneTwister(rand(orng,UInt64)), obs_rng=orng)),
        ("MCTS-ro ucb", (img,ug,orng) -> plan_mcts_rollout(img,ug,noise_sigma,(20,36,30),pi/8;
            z_update=24,transition_k=0.0, n_rollouts=200, rollout_policy=:ucb, rollout_alpha=1.0,
            sample_rng=MersenneTwister(rand(orng,UInt64)), obs_rng=orng)),
        ("MCTS-ro coneinfo", (img,ug,orng) -> plan_mcts_rollout(img,ug,noise_sigma,(20,36,30),pi/8;
            z_update=24,transition_k=0.0, n_rollouts=200, rollout_policy=:coneinfo, rollout_alpha=0.5,
            sample_rng=MersenneTwister(rand(orng,UInt64)), obs_rng=orng)),
    ]

    for (name, plan_fn) in mcts_policies
        vals = Float64[]
        for sim in 1:n_mcts
            img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
            ug = noise_sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
            _, lv, _, _ = plan_fn(img, ug, MersenneTwister(pseeds[sim]))
            push!(vals, lv)
        end
        ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
        @printf("  %-22s  %8.2f  %8.2f  %8.2f  %8.2f (50)\n",
            name, ent, quantile(vals,0.05), quantile(vals,0.10), mean(vals))
    end
end

run_suite(update_mode=:deterministic, label="DETERMINISTIC")
run_suite(update_mode=:altitude_weighted, label="ALTITUDE-WEIGHTED (stochastic)")
