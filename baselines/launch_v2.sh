#!/bin/bash
cd "/c/Users/Mahdi Al-Husseini/Dropbox/Mahdi Al Husseini/academics/aerospace_engineering/research/lunar_bandit/lander_site_selection/baselines"

# V2 is ~5s/sim, so 250 sims per chunk = ~1250s per worker
# Run 4 workers at a time: split each (sigma,k) into 4 chunks of 250

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

# sigma=0.5 already done from previous run, skip it

echo "=== V2 sigma=2.0 k=0 ==="
run_batch "v2 2.0 0.0 1 250" "v2 2.0 0.0 251 500" "v2 2.0 0.0 501 750" "v2 2.0 0.0 751 1000"

echo "=== V2 sigma=2.0 k=4 ==="
run_batch "v2 2.0 4.0 1 250" "v2 2.0 4.0 251 500" "v2 2.0 4.0 501 750" "v2 2.0 4.0 751 1000"

echo "=== V2 sigma=2.0 k=7 ==="
run_batch "v2 2.0 7.0 1 250" "v2 2.0 7.0 251 500" "v2 2.0 7.0 501 750" "v2 2.0 7.0 751 1000"

echo "=== V2 sigma=2.0 k=15 ==="
run_batch "v2 2.0 15.0 1 250" "v2 2.0 15.0 251 500" "v2 2.0 15.0 501 750" "v2 2.0 15.0 751 1000"

echo "=== V2 sigma=5.0 k=0 ==="
run_batch "v2 5.0 0.0 1 250" "v2 5.0 0.0 251 500" "v2 5.0 0.0 501 750" "v2 5.0 0.0 751 1000"

echo "=== V2 sigma=5.0 k=4 ==="
run_batch "v2 5.0 4.0 1 250" "v2 5.0 4.0 251 500" "v2 5.0 4.0 501 750" "v2 5.0 4.0 751 1000"

echo "=== V2 sigma=5.0 k=7 ==="
run_batch "v2 5.0 7.0 1 250" "v2 5.0 7.0 251 500" "v2 5.0 7.0 501 750" "v2 5.0 7.0 751 1000"

echo "=== V2 sigma=5.0 k=15 ==="
run_batch "v2 5.0 15.0 1 250" "v2 5.0 15.0 251 500" "v2 5.0 15.0 501 750" "v2 5.0 15.0 751 1000"

echo "=== V2 sigma=0.5 k=0 ==="
run_batch "v2 0.5 0.0 1 250" "v2 0.5 0.0 251 500" "v2 0.5 0.0 501 750" "v2 0.5 0.0 751 1000"

echo "=== V2 sigma=0.5 k=4 ==="
run_batch "v2 0.5 4.0 1 250" "v2 0.5 4.0 251 500" "v2 0.5 4.0 501 750" "v2 0.5 4.0 751 1000"

echo "=== V2 sigma=0.5 k=7 ==="
run_batch "v2 0.5 7.0 1 250" "v2 0.5 7.0 251 500" "v2 0.5 7.0 501 750" "v2 0.5 7.0 751 1000"

echo "=== V2 sigma=0.5 k=15 ==="
run_batch "v2 0.5 15.0 1 250" "v2 0.5 15.0 251 500" "v2 0.5 15.0 501 750" "v2 0.5 15.0 751 1000"

echo "=== V2 ALL DONE ==="
