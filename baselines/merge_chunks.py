"""Merge chunk CSVs into final results table."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, os, glob

beta = 0.5
def ce(v): return -(1/beta)*np.log(np.mean(np.exp(-beta*v))+1e-15)

data_dir = os.path.join(os.path.dirname(__file__), "..", "data_final")
version = "v3"

print(f"MCTS {version.upper()} — 1000 sims, CE / P01 / Mean")
print(f"{'':22s}  {'k=0':>8s}  {'k=4':>8s}  {'k=7':>8s}  {'k=15':>8s}")
print("-"*60)

for metric_name in ["CE", "P01", "Mean"]:
    print(f"\n  {metric_name}:")
    for sigma in [0.5, 2.0, 5.0]:
        sig_str = f"{sigma:.1f}".replace(".", "_")
        row = f"  sig={sigma:<4}"
        for k in [0, 4, 7, 15]:
            pattern = f"mcts_{version}_k{k}_sigma{sig_str}_sims*.csv"
            files = sorted(glob.glob(os.path.join(data_dir, pattern)))
            if not files:
                row += f"  {'N/A':>8s}"
                continue
            vals = []
            for f in files:
                vals.extend([float(x.strip()) for x in open(f) if x.strip()])
            vals = np.array(vals)
            if metric_name == "CE":
                row += f"  {ce(vals):8.2f}"
            elif metric_name == "P01":
                row += f"  {np.percentile(vals, 1):8.2f}"
            else:
                row += f"  {np.mean(vals):8.2f}"
        print(row)
