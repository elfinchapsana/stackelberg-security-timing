"""Reproduce every experiment, table number and figure of the paper.

Usage (from this folder):
    python run_all.py               # everything (about 10-25 minutes on a laptop)
    python run_all.py --skip-e5     # skip the timing study E5
    python run_all.py --workers 4   # number of CPU processes for E3/E4

Outputs: results/*.csv (raw data and comparison.csv), figures/*.png
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E, figures as F, compare as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e5", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    t0 = time.time()
    print("E1 five-asset example"); E.run_e1()
    print("E2 sensitivity"); E.run_e2()
    print("E3 random sweep (10,800 instances)"); E.run_sweep("synthetic", workers=args.workers)
    print("E4 timing-anchored scenario and sweep (5,400 instances)"); E.run_e4_anchored(); E.run_sweep("measured", workers=args.workers)
    print("Validation"); E.run_validation()
    if not args.skip_e5:
        print("E5 scalability (single core: set OMP_NUM_THREADS=1 before running for clean timings)"); E.run_e5()
    print("Figures"); F.all_figures(include_e5=not args.skip_e5 or os.path.exists("results/E5_scalability.csv"))
    print("Comparison with the numbers in the paper -> results/comparison.csv")
    df = C.build(); df.to_csv("results/comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"Finished in {time.time() - t0:.0f} s")
