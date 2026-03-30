#!/bin/bash
cd "/c/Users/Mahdi Al-Husseini/Dropbox/Mahdi Al Husseini/academics/aerospace_engineering/research/lunar_bandit/lander_site_selection/baselines"

# Run 4 Julia processes at a time, each handling 500 sims
# Total: 3 sigma x 4 k x 2 chunks = 24 jobs

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

echo "=== Batch 1: sigma=0.5, all k, first 500 ==="
run_batch "v3 0.5 0.0 1 500" "v3 0.5 4.0 1 500" "v3 0.5 7.0 1 500" "v3 0.5 15.0 1 500"

echo "=== Batch 2: sigma=0.5, all k, last 500 ==="
run_batch "v3 0.5 0.0 501 1000" "v3 0.5 4.0 501 1000" "v3 0.5 7.0 501 1000" "v3 0.5 15.0 501 1000"

echo "=== Batch 3: sigma=2.0, all k, first 500 ==="
run_batch "v3 2.0 0.0 1 500" "v3 2.0 4.0 1 500" "v3 2.0 7.0 1 500" "v3 2.0 15.0 1 500"

echo "=== Batch 4: sigma=2.0, all k, last 500 ==="
run_batch "v3 2.0 0.0 501 1000" "v3 2.0 4.0 501 1000" "v3 2.0 7.0 501 1000" "v3 2.0 15.0 501 1000"

echo "=== Batch 5: sigma=5.0, all k, first 500 ==="
run_batch "v3 5.0 0.0 1 500" "v3 5.0 4.0 1 500" "v3 5.0 7.0 1 500" "v3 5.0 15.0 1 500"

echo "=== Batch 6: sigma=5.0, all k, last 500 ==="
run_batch "v3 5.0 0.0 501 1000" "v3 5.0 4.0 501 1000" "v3 5.0 7.0 501 1000" "v3 5.0 15.0 501 1000"

echo "=== ALL DONE ==="
