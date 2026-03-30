using Printf, Statistics
include(joinpath(@__DIR__, "..", "environment.jl"))
include("ucb.jl"); include("lcb.jl"); include("greedy_risk.jl")

function run_sweep()
    n_sims = 500; seed = 1234; beta = 0.5
    global GLOBAL_UPDATE_MODE = :altitude_weighted
    global GLOBAL_DECAY_LAMBDA = 0.6

    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    ks = [0.0, 2.0, 4.0, 7.0, 15.0]

    policy_defs = [
        ("BayesUCB a=0.5",  (ns,orng) -> plan_ucb(img,ug,ns,(20,36,30),pi/8; z_update=24,transition_k=tk, α=0.5, obs_rng=orng)),
        ("BayesLCB a=1.0",  (ns,orng) -> plan_lcb(img,ug,ns,(20,36,30),pi/8; z_update=24,transition_k=tk, α=1.0, obs_rng=orng)),
        ("EVaR a=0.01",     (ns,orng) -> plan_greedy_risk(img,ug,ns,(20,36,30),pi/8; z_update=24,transition_k=tk, risk_cfg=RiskConfig(mode=RiskEVaR,alpha=0.01), obs_rng=orng)),
        ("Max sig",         (ns,orng) -> plan_greedy_risk(img,ug,ns,(20,36,30),pi/8; z_update=24,transition_k=tk, risk_cfg=RiskConfig(mode=RiskEntropicSigma,beta=0.5,sigma_ref_mode=SigmaMax), obs_rng=orng)),
        ("CVaR a=0.10",     (ns,orng) -> plan_greedy_risk(img,ug,ns,(20,36,30),pi/8; z_update=24,transition_k=tk, risk_cfg=RiskConfig(mode=RiskCVaR,alpha=0.10), obs_rng=orng)),
        ("Max sig +Expl",   (ns,orng) -> plan_greedy_risk(img,ug,ns,(20,36,30),pi/8; z_update=24,transition_k=tk, risk_cfg=RiskConfig(mode=RiskEntropicSigma,beta=0.5,sigma_ref_mode=SigmaMax), λ_explore=0.1, obs_rng=orng)),
    ]

    # For each sigma, print a heatmap-style table: policies × k
    for sigma in sigmas
        # Pre-generate seeds
        rng = MersenneTwister(seed); rng_p = MersenneTwister(seed + 2000)
        tseeds = Int[]; nseeds = Int[]; pseeds = Int[]
        for _ in 1:n_sims
            push!(tseeds, rand(rng, 1:10^9))
            push!(nseeds, rand(rng, 1:10^9))
            push!(pseeds, rand(rng_p, 1:10^9))
        end

        println("\n", "="^75)
        @printf("  σ = %.1f  (altitude-weighted, 500 sims, Entropic metric)\n", sigma)
        println("="^75)
        @printf("  %-18s", "Policy")
        for k in ks; @printf("  %7s", "k=$(Int(k))"); end
        println()
        println("  ", "-"^70)

        for (pname, _) in policy_defs
            @printf("  %-18s", pname)
            for tk in ks
                # Need to capture tk and sigma in closures properly
                vals = Float64[]
                for sim in 1:n_sims
                    img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
                    ug = sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
                    orng = MersenneTwister(pseeds[sim])

                    _, lv, _, _ = if pname == "BayesUCB a=0.5"
                        plan_ucb(img, ug, sigma, (20,36,30), pi/8; z_update=24, transition_k=tk, α=0.5, obs_rng=orng)
                    elseif pname == "BayesLCB a=1.0"
                        plan_lcb(img, ug, sigma, (20,36,30), pi/8; z_update=24, transition_k=tk, α=1.0, obs_rng=orng)
                    elseif pname == "EVaR a=0.01"
                        plan_greedy_risk(img, ug, sigma, (20,36,30), pi/8; z_update=24, transition_k=tk, risk_cfg=RiskConfig(mode=RiskEVaR,alpha=0.01), obs_rng=orng)
                    elseif pname == "Max sig"
                        plan_greedy_risk(img, ug, sigma, (20,36,30), pi/8; z_update=24, transition_k=tk, risk_cfg=RiskConfig(mode=RiskEntropicSigma,beta=0.5,sigma_ref_mode=SigmaMax), obs_rng=orng)
                    elseif pname == "CVaR a=0.10"
                        plan_greedy_risk(img, ug, sigma, (20,36,30), pi/8; z_update=24, transition_k=tk, risk_cfg=RiskConfig(mode=RiskCVaR,alpha=0.10), obs_rng=orng)
                    else # Max sig +Expl
                        plan_greedy_risk(img, ug, sigma, (20,36,30), pi/8; z_update=24, transition_k=tk, risk_cfg=RiskConfig(mode=RiskEntropicSigma,beta=0.5,sigma_ref_mode=SigmaMax), λ_explore=0.1, obs_rng=orng)
                    end
                    push!(vals, lv)
                end
                ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
                @printf("  %7.2f", ent)
            end
            println()
        end
    end
end

run_sweep()
