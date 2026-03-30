using Printf, Statistics
include(joinpath(@__DIR__, "..", "environment.jl"))
include("mcts_tree.jl")
include("mcts_variants.jl")

function run_full_sweep()
    n_sims = 50; seed = 1234; beta = 0.5
    global GLOBAL_UPDATE_MODE = :deterministic
    set_v4_window!(2)

    rng = MersenneTwister(seed); rng_p = MersenneTwister(seed + 2000)
    tseeds = Int[]; nseeds = Int[]; pseeds = Int[]
    for _ in 1:n_sims
        push!(tseeds, rand(rng, 1:10^9))
        push!(nseeds, rand(rng, 1:10^9))
        push!(pseeds, rand(rng_p, 1:10^9))
    end

    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    ks = [0.0, 2.0, 4.0, 7.0, 15.0]
    versions = [
        ("UCT V1 (oracle)",     :v1),
        ("UCT V2 (known dyn)",  :v2),
        ("UCT V3 (frozen)",     :v3),
        ("UCT V4 (learned)",    :v4),
    ]

    for sigma in sigmas
        println("\n", "="^70)
        @printf("  σ = %.1f  (deterministic, 50 sims, 700 iter, c=3.0, Entropic)\n", sigma)
        println("="^70)
        @printf("  %-22s", "Variant")
        for k in ks; @printf("  %7s", "k=$(Int(k))"); end
        println()
        println("  ", "-"^60)

        for (name, version) in versions
            @printf("  %-22s", name)
            for tk in ks
                vals = Float64[]
                for sim in 1:n_sims
                    img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
                    ug = sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
                    _, lv, _, _ = plan_mcts_uct(img, ug, sigma, (20,36,30), pi/8;
                        z_update=24, transition_k=tk, version=version,
                        iterations=700, exploration_c=3.0,
                        rng=MersenneTwister(pseeds[sim]), obs_rng=MersenneTwister(pseeds[sim]))
                    push!(vals, lv)
                end
                ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
                @printf("  %7.2f", ent)
            end
            println()
            flush(stdout)
        end
    end
end
run_full_sweep()
