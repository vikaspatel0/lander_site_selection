#!/bin/bash
cd "/c/Users/Mahdi Al-Husseini/Dropbox/Mahdi Al Husseini/academics/aerospace_engineering/research/lunar_bandit/lander_site_selection/baselines"

run_batch() {
    local pids=()
    for args in "$@"; do
        julia run_mcts_chunk.jl $args &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid
    done
}

for version in v3_cvar v3_maxsig; do
    for sigma in 0.5 2.0 5.0; do
        echo "=== $version sigma=$sigma ==="
        run_batch "$version $sigma 0.0 1 500" "$version $sigma 4.0 1 500" "$version $sigma 7.0 1 500" "$version $sigma 15.0 1 500"
        run_batch "$version $sigma 0.0 501 1000" "$version $sigma 4.0 501 1000" "$version $sigma 7.0 501 1000" "$version $sigma 15.0 501 1000"
    done
done

echo "=== V3 RISK VARIANTS ALL DONE ==="
