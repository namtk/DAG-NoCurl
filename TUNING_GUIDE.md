# Tuning guide for DAG-NoCurl

This guide explains the main parameters to be tuned when running the code in this repository, how each parameter affects results, and practical strategies to tune them depending on input data regime (small-n / large-d, noisy data, etc.). It also gives reproducible tuning recipes (grid search, cross-validation, stability selection) and example commands.

## Quick overview of important parameters

- `--data_sample_size` (n): number of observations/samples. Larger n improves statistical power.
- `--data_variable_size` (d): number of variables / nodes. Larger d increases problem difficulty and runtime (optimizers scale roughly with d^2 operations).
- `--graph_type`, `--graph_degree`, `--graph_sem_type`: data-generation choices (only relevant if generating synthetic data with `--generate_data=1`).
- `--methods` (`nocurl`, `notear`, `CAM`, `GES`, `MMPC`, `FGS`): algorithm to run. `nocurl` is the authors' NoCurl method; `notear` is original NOTEARS variant; others are baselines requiring extra deps.
- `--lambda1`, `--lambda2`: regularization / penalty coefficients used in the two-stage AL / projection routines in `BPR.fit_*`.
- `--graph_threshold`: threshold used to binarize the learned weighted adjacency matrix into a graph (default 0.3). Higher values make the final graph sparser.
- `--h_tol`: tolerance for the acyclicity objective h(A) used to stop the augmented-Lagrangian loops (defaults around 1e-8). Smaller tolerances force a more exact acyclicity but increase runtime.
- `--rho_A_max`: maximum multiplier for the augmented-Lagrangian (large value prevents unbounded increase in rho).
- `--train_epochs`: upper bound on optimization iterations (used in notear-like routines). Larger values let optimizer run longer.
- `--repeat`: how many experiment repeats (useful for averaging metrics). More repeats yield stable estimates of performance at the cost of runtime.
- `--generate_data`: whether to generate synthetic data (1) or expect precomputed pickles (0).

Lower-level notes (inside `BPR.py`): L-BFGS-B is used for continuous optimization; the code also internally thresholds small weights to zero before reporting nnz.

## Effects and rules-of-thumb

- Lambda1 / Lambda2 (regularization strengths):
  - Effect: larger lambda increases penalty on h(A) or other regularizers; typically leads to more strongly enforced acyclic solutions.
  - Rule-of-thumb:
    - If n is small relative to d (n << d), increase regularization (try lambda in larger range) to avoid overfitting and dense spurious edges.
    - Try exponentially spaced values: lambda in {1, 10, 100, 1000, 10000} (both for lambda1 and lambda2). Often the README example uses lambda1=10, lambda2=1000.

- `--graph_threshold` (binarize learned weights):
  - Effect: directly changes the number of reported edges `nnz` and therefore SHD/tpr/fdr.
  - Rule-of-thumb: search thresholds in [0.1, 0.2, 0.3, 0.5]. If having a ground-truth expected edge count, choose the threshold that results in nnz closest to that count.

- `--h_tol` (acyclicity tolerance):
  - Effect: smaller tolerance forces the optimizer to reduce h(A) further (more exact DAGness). This can increase runtime and may slightly change the final weights.
  - Rule-of-thumb: default 1e-8 is strict. For fast prototyping use 1e-6 or 1e-5; for final runs keep 1e-8 or lower.

- `--train_epochs`:
  - Effect: maximum iterations of the Notears-style solver. If optimization hasn't converged, raise this value.

- `--repeat`:
  - Use `repeat >= 10` to compute mean/std for robust evaluation. For quick debugging set `repeat=1`.

## Tuning strategies

1) Grid search (brute force)

 - When having ground-truth (synthetic) data, grid search is straightforward: run the algorithm across combinations of `lambda1`, `lambda2` and `graph_threshold`, compute the averaged evaluation metric (e.g., SHD, tpr, fdr) and choose the combination with lowest SHD or a good precision/recall trade-off.

Example shell snippet to sweep lambda1/lambda2 (small example):

```bash
source .venv/bin/activate
for L1 in 1 10 100; do
  for L2 in 10 100 1000; do
    python main_efficient.py --data_sample_size=200 --data_variable_size=10 --repeat=5 \
      --methods=nocurl --generate_data=1 --lambda1=$L1 --lambda2=$L2 --h_tol=1e-8
  done
done
```

After the runs, inspect `results/*.pkl` to find average SHD and pick the best parameters.

2) Cross-validation using predictive loss (no ground truth)

 - Split X into train/validation (e.g., 80/20). For each parameter combo, fit on train and compute `utils.get_loss_L2(W_est, X_val, 'l2')` (or equivalent predictive score) on hold-out data. Choose parameters minimizing validation loss. This favors models that generalize in reconstructing the data distribution.

Python pseudo-code:

```python
from sklearn.model_selection import train_test_split
from BPR import BPR
import numpy as np

X_train, X_val = train_test_split(X, test_size=0.2)
best = None
for L1 in [1,10,100]:
    for L2 in [10,100,1000]:
        args.lambda1, args.lambda2 = float(L1), float(L2)
        est = BPR(args)
        W, *_ = est.fit(X_train, method='nocurl')
        loss_val, _ = utils.get_loss_L2(W, X_val, 'l2')
        # store loss_val and pick smallest
```

## Tuning by data regime

- Small n, large d (n << d):
  - Increase lambda (more regularization) to avoid spurious edges.
  - Use stronger binarization thresholds (e.g., > 0.3).
  - Use cross-validated predictive loss cautiously — high variance may mislead.

- Large n, moderate d (n >> d):
  - Tolerate smaller lambda (less regularization) and reduce `graph_threshold` to recover weaker edges.
  - Use smaller `h_tol` (e.g., 1e-8) for more precise DAG enforcement.

- Noisy data (high measurement noise):
  - Increase regularization. Also consider pre-processing (denoising or PCA) if appropriate.

- Nonlinear / misspecified model: if the true SEM is nonlinear but the linear solver is used, expect degraded SHD/tpr. Consider switching to nonlinear baselines or using the `data_type=nonlinear*` code paths and appropriate baseline methods.

## Practical tips and diagnostics

- Monitor the acyclicity objective h(A): if h(A) is not decreasing or stalls far above `h_tol`, consider increasing `rho_A_max` or re-initializing weights; check logs for L-BFGS warnings.
- If the solver returns very dense W: increase lambda values and/or increase `graph_threshold`.
- If estimates are unstable across seeds: use `--repeat` with different random seeds to extract robust edges.
- When comparing many parameter combos, store `results/*.pkl` and write a small script to aggregate and plot SHD / nnz vs parameters to find sweet spots.

## Example workflow for robust tuning

1. Quick coarse grid: lambda1 in {1,10,100}, lambda2 in {10,100,1000}, graph_threshold in {0.1,0.3} with repeat=3.
2. Inspect results and pick a narrow promising sub-range.
4. Optionally validate using held-out predictive loss.

## References in code

- See `main_efficient.py` for CLI args parsing and where `lambda1`, `lambda2`, `graph_threshold`, `h_tol`, `train_epochs`, `generate_data` are used.
- See `BPR.py` for implementation details of the optimization routines.
- See `utils.get_loss_L2` in `utils.py` for predictive loss evaluation.

