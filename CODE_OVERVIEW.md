# Code overview — DAG-NoCurl

This file briefly explains the key source files in this repository and the main implementation pieces (e.g., where the algorithm is implemented, how data is generated, and where evaluation is performed).

## Top-level files

- `main_efficient.py`
  - Entry point for experiments. Parses CLI args, generates or loads datasets, calls the solver (`BPR.BPR`) and evaluates results.
  - Key flow:
    1. Parse CLI args (see `get_args`).
    2. For each trial: generate/load data pickle `./data/lineardata/{n}_{d}_{graph_type}_{degree}_{sem}_{trial}.pkl` or load nonlinear data paths.
    3. Create a `BPR.BPR` object and call `bpr.fit(X, method)` where `method` determines algorithm.
    4. Convert outputs into a NetworkX graph, call `utils.count_accuracy_new` to compute metrics, and write results with `utils.print_to_file`.

- `README.md`, `RUN_REPORT.md`, `TUNING_GUIDE.md`
  - High-level docs, quick-run example, and tuning guidance (the latter two were added during this session).

## Core algorithm implementation

- `BPR.py` — central implementation of the solver(s).
  - Class `BPR` wraps different fitting methods under `fit(self, X, method)`.
  - Main implemented methods:
    - `fit_all` — original NOTEARS-like augmented Lagrangian routine (L-BFGS-B inner solver). Uses `_h(w)` to compute the acyclicity measure and its gradient; iteratively increases `rho` and updates `alpha` until `h <= h_tol`.
    - `fit_all_L2proj` — NoCurl-style method (the code labels it L2 projection / AL2proj). It calls `fit_aug_lagr_AL2proj` which contains a two-stage optimization:
       - a pre-optimization (`_prefunc`, `_pregrad`) to get an initial W, then
       - a structured optimization over variables `w` and `phi` using L-BFGS-B (`sopt.minimize`) to obtain `W` that is enforced to be acyclic via Hodge decomposition.
    - `fit_cam`, `fit_mmpc`, `fit_ges`, `fit_fgs` — wrappers to external baselines (they convert outputs of `cdt` or `pycausal` into numpy adjacency matrices and compute predictive losses).
  - Thresholding: after optimization `W` has small continuous weights; the code sets small values to zero and also allows a CLI `--graph_threshold` to binarize the final adjacency for evaluation.

## Utilities and evaluation

- `utils.py`
  - Data generation:
    - `simulate_random_dag(d, degree, graph_type, w_range)` — creates a random weighted DAG using an adjacency skeleton (Erdos–Rényi, Barabási–Albert, full, chain). It permutes nodes and samples edge weights in `w_range` with random signs.
    - `simulate_sem` / `simulate_sem_multid` / `simulate_sem_nonlinear` — sample data X from the SEM defined by W and specified `sem_type` (linear-gauss, linear-exp, linear-gumbel) or nonlinear variants.
  - Metrics / evaluation:
    - `count_accuracy`, `count_accuracy_new` — compute fdr, tpr, fpr, SHD, nnz, and detailed error counts (extra/missing/reverse).
    - `get_loss_L2` — compute predictive loss and gradient for L2 regression loss; used in evaluation and in objective terms.
  - IO / logging:
    - `print_to_file` — writes a `.txt` summary and pickled DataFrame (`results/... .pkl`) that contains per-trial metrics.
    - `setup_logger` — simple logger wrapper.

## FGES helper

- `fges_continuous_yyu.py`
  - Helper code to run FGES via Tetrad/pycausal (Java). It runs Tetrad via `pycausal` calls, converts a dot result into an adjacency using `pydot`, and returns an adjacency matrix.
  - Requires `pycausal` and Java; not required for NoCurl but present for baseline comparisons.

## Data and results layout

- Generated datasets: `data/lineardata/{n}_{d}_{graph_type}_{degree}_{sem}_{trial}.pkl` — each file contains `(G, X)` where `G` is a NetworkX DiGraph and `X` is a numpy array of samples.
- Results: `results/{...}.txt` and `results/{...}.pkl` (pickled pandas DataFrame). The DataFrame columns are `['time','lossW','SHD','nnz','tpr','fpr','fdr','h','extra','missing','reverse']`.

## Running baselines and optional dependencies

- Baselines (`CAM`, `GES`, `MMPC`) use `cdt` (Causal Discovery Toolbox) — installing `cdt` may require extra system packages and optionally R.
- FGES requires `pycausal` and Java/Tetrad jars and Graphviz (`pydot`) for parsing dot outputs.

## Where to change behavior / tune

- `main_efficient.py` holds CLI defaults (lambda1, lambda2, graph_threshold, h_tol, train_epochs, generate_data, etc.).
- `BPR.py` contains the solver internals; tune lines inside `fit_aug_lagr_AL2proj` / `_prefunc` / `_grad` for algorithmic modifications.
- `utils.py` contains all data generation and evaluation functions — modify them to add new metrics or synthetic SEM types.




