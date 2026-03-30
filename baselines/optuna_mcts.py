"""
Optuna hyperparameter optimization for UCT MCTS V4.
Optimizes iterations and exploration_c to maximize mean Entropic
across multiple (sigma, k) configurations.
"""

import optuna
import subprocess
import re
import sys

N_SIMS = 30  # per config during search (fast)
CONFIGS = [(5.0, 0.0), (5.0, 4.0), (5.0, 15.0), (3.0, 0.0)]
BETA = 0.5
BASELINES_DIR = __file__.replace("optuna_mcts.py", "")


def run_trial(iterations: int, exploration_c: float) -> float:
    """Run UCT V4 across all configs, return mean Entropic."""

    julia_code = f"""
    using Printf, Statistics
    include(joinpath(@__DIR__, "..", "environment.jl"))
    include("mcts_tree.jl")
    include("mcts_variants.jl")

    function run_configs()
        n_sims = {N_SIMS}; seed = 1234; beta = {BETA}
        global GLOBAL_UPDATE_MODE = :deterministic
        set_v4_window!(2)

        rng = MersenneTwister(seed); rng_p = MersenneTwister(seed + 2000)
        tseeds = Int[]; nseeds = Int[]; pseeds = Int[]
        for _ in 1:n_sims
            push!(tseeds, rand(rng, 1:10^9))
            push!(nseeds, rand(rng, 1:10^9))
            push!(pseeds, rand(rng_p, 1:10^9))
        end

        configs = {[(s, k) for s, k in CONFIGS]}
        entropics = Float64[]

        for (sigma, tk) in configs
            vals = Float64[]
            for sim in 1:n_sims
                img = generate_terrain_2(60; seed=tseeds[sim], value_min=0.0, value_range=10.0)
                ug = sigma .* randn(MersenneTwister(nseeds[sim]), 60, 60)
                _, lv, _, _ = plan_mcts_uct(img, ug, sigma, (20,36,30), pi/8;
                    z_update=24, transition_k=tk, version=:v4,
                    iterations={iterations}, exploration_c={exploration_c},
                    rng=MersenneTwister(pseeds[sim]), obs_rng=MersenneTwister(pseeds[sim]))
                push!(vals, lv)
            end
            ent = -(1.0/beta)*log(mean(exp.(-beta.*vals))+eps())
            push!(entropics, ent)
        end

        mean_ent = mean(entropics)
        @printf("RESULT: %.4f\\n", mean_ent)
        for (idx, (sigma, tk)) in enumerate(configs)
            @printf("  sigma=%.1f k=%.0f: %.2f\\n", sigma, tk, entropics[idx])
        end
    end
    run_configs()
    """

    result = subprocess.run(
        ["julia", "-e", julia_code],
        capture_output=True, text=True, timeout=600,
        cwd=BASELINES_DIR
    )

    if result.returncode != 0:
        print(f"  Julia error: {result.stderr[-200:]}", file=sys.stderr)
        return float("-inf")

    # Parse RESULT line
    match = re.search(r"RESULT: ([-\d.]+)", result.stdout)
    if match:
        val = float(match.group(1))
        # Print per-config breakdown
        for line in result.stdout.strip().split("\n"):
            if line.strip().startswith("sigma="):
                print(f"    {line.strip()}")
        return val
    else:
        print(f"  Could not parse output: {result.stdout[-200:]}", file=sys.stderr)
        return float("-inf")


def objective(trial: optuna.Trial) -> float:
    iterations = trial.suggest_int("iterations", 100, 5000, log=True)
    exploration_c = trial.suggest_float("exploration_c", 0.1, 5.0, log=True)

    print(f"\nTrial {trial.number}: iterations={iterations}, c={exploration_c:.3f}")
    score = run_trial(iterations, exploration_c)
    print(f"  -> mean Entropic = {score:.4f}")
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="mcts_v4_hpopt",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    n_trials = 30
    print(f"Running {n_trials} Optuna trials...")
    print(f"Configs: {CONFIGS}")
    print(f"Sims per config: {N_SIMS}")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("Best trial:")
    print(f"  iterations = {study.best_params['iterations']}")
    print(f"  exploration_c = {study.best_params['exploration_c']:.4f}")
    print(f"  mean Entropic = {study.best_value:.4f}")

    print("\nTop 5 trials:")
    for t in sorted(study.trials, key=lambda t: t.value if t.value else float("-inf"), reverse=True)[:5]:
        print(f"  iter={t.params['iterations']}, c={t.params['exploration_c']:.3f} -> {t.value:.4f}")
