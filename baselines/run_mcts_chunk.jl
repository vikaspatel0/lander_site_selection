# Usage: julia run_mcts_chunk.jl <version> <sigma> <k> <start_sim> <end_sim>
# Runs a chunk of sims for one (version, sigma, k) config, writes results to CSV

using Printf, Statistics
include(joinpath(@__DIR__, "..", "environment.jl"))
include("mcts_tree.jl")
include("mcts_variants.jl")

function main()
    version = Symbol(ARGS[1])
    sigma = parse(Float64, ARGS[2])
    tk = parse(Float64, ARGS[3])
    sim_start = parse(Int, ARGS[4])
    sim_end = parse(Int, ARGS[5])

    seed = 1234; beta = 0.5
    global GLOBAL_UPDATE_MODE = :deterministic

    # Generate all 1000 seeds (same as always)
    rng = MersenneTwister(seed); rng_p = MersenneTwister(seed + 2000)
    tseeds = Int[]; nseeds = Int[]; pseeds = Int[]
    for _ in 1:1000
        push!(tseeds, rand(rng, 1:10^9))
        push!(nseeds, rand(rng, 1:10^9))
        push!(pseeds, rand(rng_p, 1:10^9))
    end

    vals = Float64[]
    for sim in sim_start:sim_end
        img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
        ug = sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
        _, lv, _, _ = plan_mcts_uct(img, ug, sigma, (20,36,30), pi/8;
            z_update=24, transition_k=tk, version=version,
            iterations=700, exploration_c=3.0,
            rng=MersenneTwister(pseeds[sim]), obs_rng=MersenneTwister(pseeds[sim]))
        push!(vals, lv)
    end

    # Write chunk results to file
    k_str = replace(string(Int(tk)), "." => "")
    sig_str = replace(@sprintf("%.1f", sigma), "." => "_")
    outfile = joinpath(@__DIR__, "..", "data_final",
        "mcts_$(version)_k$(k_str)_sigma$(sig_str)_sims$(sim_start)_$(sim_end).csv")
    open(outfile, "w") do io
        for v in vals
            println(io, v)
        end
    end

    ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
    @printf("DONE %s sigma=%.1f k=%.0f sims=%d-%d CE=%.4f Mean=%.4f n=%d\n",
        version, sigma, tk, sim_start, sim_end, ent, mean(vals), length(vals))
end
main()
