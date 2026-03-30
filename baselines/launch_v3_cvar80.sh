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

for sigma in 0.5 2.0 5.0; do
    echo "=== v3_cvar80 sigma=$sigma ==="
    run_batch "v3_cvar80 $sigma 0.0 1 500" "v3_cvar80 $sigma 4.0 1 500" "v3_cvar80 $sigma 7.0 1 500" "v3_cvar80 $sigma 15.0 1 500"
    run_batch "v3_cvar80 $sigma 0.0 501 1000" "v3_cvar80 $sigma 4.0 501 1000" "v3_cvar80 $sigma 7.0 501 1000" "v3_cvar80 $sigma 15.0 501 1000"
done

echo "=== V3 CVAR80 ALL DONE ==="
