# Usage: julia run_mcts_single.jl <version> <sigma>
# Runs one MCTS version across all k values, 1000 sims, saves CSV results

using Printf, Statistics
include(joinpath(@__DIR__, "..", "environment.jl"))
include("mcts_tree.jl")
include("mcts_variants.jl")

function main()
    version = Symbol(ARGS[1])  # :v1, :v2, :v3
    sigma = parse(Float64, ARGS[2])
    
    n_sims = 1000; seed = 1234; beta = 0.5
    global GLOBAL_UPDATE_MODE = :deterministic

    rng = MersenneTwister(seed); rng_p = MersenneTwister(seed + 2000)
    tseeds = Int[]; nseeds = Int[]; pseeds = Int[]
    for _ in 1:n_sims
        push!(tseeds, rand(rng, 1:10^9))
        push!(nseeds, rand(rng, 1:10^9))
        push!(pseeds, rand(rng_p, 1:10^9))
    end

    ks = [0.0, 4.0, 7.0, 15.0]

    for tk in ks
        vals = Float64[]
        t = @elapsed for sim in 1:n_sims
            img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
            ug = sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
            _, lv, _, _ = plan_mcts_uct(img, ug, sigma, (20,36,30), pi/8;
                z_update=24, transition_k=tk, version=version,
                iterations=700, exploration_c=3.0,
                rng=MersenneTwister(pseeds[sim]), obs_rng=MersenneTwister(pseeds[sim]))
            push!(vals, lv)
        end
        ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
        p01 = quantile(vals, 0.01)
        mn = mean(vals)
        @printf("RESULT %s sigma=%.1f k=%.0f CE=%.4f P01=%.4f Mean=%.4f time=%.0fs\n",
            version, sigma, tk, ent, p01, mn, t)
        flush(stdout)
    end
end
main()
