# Stackelberg Security Resource Allocation Under Shared Real-Time Constraints: Equilibrium Analysis and the Price of Decoupling - CODE \& DATA

This repository contains the code and data for the paper *"Stackelberg Security Resource Allocation Under Shared Real-Time Constraints: Equilibrium Analysis and the Price of Decoupling"* (submitted to the European Journal of Operational Research).

It regenerates every experiment (E1 to E5), every number quoted in the paper, and Figures 1 to 8. The `results/` folder holds the raw outputs of the run reported in the paper.

## Install (once)

```
python -m pip install -r requirements.txt
```

Python 3.10 or newer.

## Run everything (Windows Command Prompt, from this folder)

```
set OMP\_NUM\_THREADS=1
set OPENBLAS\_NUM\_THREADS=1
python run\_all.py
```

This takes roughly 10 to 25 minutes on a laptop. Plug in the charger and close other programs, especially if you want clean E5 timings.

Options:

* `python run\_all.py --skip-e5` skips the timing study, if you only want E1 to E4.
* `python run\_all.py --workers 4` sets how many CPU cores E3/E4 may use. The default is all cores.

## Files

|File|Contents|
|-|-|
|`solver.py`|Algorithm 1 for all timing models (A, B, B', no timing), the epsilon-margin variant, minimax, attacker evaluation, cap rules (equal, delay-proportional, risk-weighted, oracle)|
|`experiments.py`|E1 to E5 and the solver validation|
|`figures.py`|Figures 1 to 8 (`figures/\*.png`, same file names as in the paper's LaTeX folder)|
|`compare.py`|Paper number against reproduced number, written to `results/comparison.csv`|
|`run\_all.py`|Runs everything in order|

## Design details

1. **E3/E4 generator.** Every loop contains at least two assets; memberships are redrawn until this holds. Without this rule, single-asset loops make equal splitting harmless, and the "every timing-binding instance has PoD > 0" result fails.
2. **E4 hardware case.** Both the PLC and the servo use the 2.6 us/frame hardware engine. The SCADA and HMI stay at 10 us, and the switch at 0.21 us.
3. **E2 grids.** Budget values are 25 log-spaced points in \[1, 40]; slack scales are 23 log-spaced points in \[0.05, 8].

All random instances are generated from fixed seeds, so every run gives the same losses. Only the E5 runtimes depend on the computer.

## How to cite

If you use this code, please cite the paper (citation to be added after publication) and this repository (DOI to be added).

## License

MIT. See `LICENSE`.

