## Quick run & report — DAG-NoCurl

This short report explains how to run the code in this repository, how the synthetic data is generated and saved, and summarizes the reconstructed graph produced by a short test run that was executed in this workspace.

---

## 1) Minimal environment and install (macOS zsh)

Create and activate a virtual environment, upgrade pip, and install the minimal required Python packages for the core path (NoCurl / NOTEARS):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy networkx pandas tqdm pydot
```

Notes:
- `pydot` is optional but useful if later running FGES which produces dot output.

## 2) Run a short synthetic test (already executed in this workspace)

The following command runs a small synthetic linear-Gaussian experiment using the NoCurl method. This is the exact command used for the quick test in the repo:

```bash
source .venv/bin/activate
python main_efficient.py \
  --data_sample_size=200 \
  --data_variable_size=6 \
  --repeat=1 \
  --methods=nocurl \
  --generate_data=1 \
  --graph_type=erdos-renyi \
  --graph_degree=2 \
  --lambda1=10 --lambda2=1000 \
  --h_tol=1e-8
```

What this does:
- Generates a random DAG (Erdos–Rényi style) with d=6 nodes and expected degree=2.
- Simulates n=200 samples from a linear Gaussian SEM (noise added per node).
- Runs the NoCurl estimator (implemented in `BPR.py`).
- Saves results in `results/` and the generated dataset in `data/lineardata/`.

## 3) Where the generated data is saved and its format

- Data pickle path used for the short run:
  - `data/lineardata/200_6_erdos-renyi_2_linear-gauss_0.pkl`

- What is inside that pickle file (as produced by `main_efficient.py`):
  - A tuple `(G, X)` saved using `pickle.dump((G, X), f)`, where:
    - `G` is a NetworkX DiGraph containing the ground-truth weighted adjacency matrix (weights on edges).
    - `X` is a numpy array of shape `[n, d]` (for these linear experiments) containing the simulated samples.

The file can be loaded in Python like this:

```python
import pickle
with open('data/lineardata/200_6_erdos-renyi_2_linear-gauss_0.pkl','rb') as f:
    G, X = pickle.load(f)

# ground-truth adjacency matrix
import networkx as nx
W_true = nx.to_numpy_array(G)
print('W_true shape:', W_true.shape)
```

## 4) Where results are written

- The script writes a human-readable summary and a `.pkl` DataFrame to `results/`. Example files created in the quick run:
  - `results/1nocurl_6_synthetic_200_erdos-renyi_linear-gauss_2_hTol_1e-08_lambda_10.0_1000.0.txt`
  - `results/1nocurl_6_synthetic_200_erdos-renyi_linear-gauss_2_hTol_1e-08_lambda_10.0_1000.0.pkl`

- The `.pkl` produced by `utils.print_to_file` is a pickled pandas DataFrame with columns (per-trial):
  - `['time','lossW', 'SHD', 'nnz','tpr','fpr','fdr','h','extra','missing','reverse']`

To inspect the saved DataFrame:

```python
import pandas as pd
df = pd.read_pickle('results/1nocurl_6_synthetic_200_erdos-renyi_linear-gauss_2_hTol_1e-08_lambda_10.0_1000.0.pkl')
print(df.columns)
print(df.iloc[0].to_dict())
```

## 5) Metrics interpretation (what to look for)

- SHD (Structural Hamming Distance): number of edge insertions/deletions/orientation flips required to convert the estimated graph to the true graph. Lower is better; 0 = exact match.
- tpr (true positive rate / recall): fraction of true edges recovered. Higher is better (1 = all true edges found).
- fdr (false discovery rate): fraction of predicted edges that are false positives. Lower is better (0 = no false discoveries).
- fpr (false positive rate): fraction of negative edges predicted as positive.
- nnz: number of nonzero entries (predicted edges) in the estimated adjacency.

Focus on SHD / tpr / fdr when judging structure recovery.

## 6) Reconstructed graph from the quick run (summary)

Re-ran the estimator on the same saved dataset and extracted the estimated adjacency. Here is the summary of the reconstructed graph from that run:

- Nodes: 6 (adjacency matrix shape: 6 x 6)
- Edges (nonzero after thresholding at 0.3): 4
- Edge list (i -> j : weight):
  - 0 -> 3 : 0.615958776645638
  - 0 -> 5 : -1.205518217271231
  - 1 -> 2 : 1.1935194924343897
  - 3 -> 5 : 1.0861013480203363

This matches the `nnz = 4` value saved in the run's DataFrame and the evaluation reported in the `.txt` summary: SHD = 0, tpr = 1.0, fdr = 0.0, fpr = 0.0.

To load the estimated adjacency programmatically (thresholded at the same default `--graph_threshold=0.3`):

```python
import pickle, numpy as np
from BPR import BPR

with open('data/lineardata/200_6_erdos-renyi_2_linear-gauss_0.pkl','rb') as f:
    G, X = pickle.load(f)

# recreate args namespace like the run
import argparse
args = argparse.Namespace()
args.rho_A_max = 1e+16
args.h_tol = 1e-8
args.lambda1 = 10.0
args.lambda2 = 1000.0
args.train_epochs = 10000
args.graph_threshold = 0.3

est = BPR(args)
A, h, alpha, rho = est.fit(X, method='nocurl')
A = np.array(A)
A_bin = (np.abs(A) > args.graph_threshold).astype(int)
print('Estimated edges (i,j,weight):')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if A_bin[i,j]:
            print(f'{i} -> {j} : {A[i,j]}')
```


