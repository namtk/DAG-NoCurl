# DAG-NoCurl

Code for DAG-NoCurl work

## Local changes and added files (this clone)

This repository was cloned from the original DAG-NoCurl project and the following files were
added or created in this working copy to help with reproducible runs, tuning, and code navigation.

New files added in this workspace:

- `RUN_REPORT.md` — quick run report describing how the repository was executed here, where generated data and results are stored, and a short summary of the reconstructed graph produced by a short test run.
- `TUNING_GUIDE.md` — parameter tuning guide: explanation of all relevant CLI parameters, rules-of-thumb for different data regimes, and tuning strategies (grid search, cross-validation).
- `CODE_OVERVIEW.md` — short code map describing key files (`main_efficient.py`, `BPR.py`, `utils.py`, `fges_continuous_yyu.py`) and where to change behavior.

Other workspace changes made while preparing reproducible runs (not all are committed):

- Created `data/lineardata/` and `results/` directories to store generated datasets and output metrics. These directories are data artifacts and typically should be created on the target machine rather than tracked in Git.
- A local virtual environment `.venv/` was used during testing; virtual environments are not committed.

What changed in code / documentation:

- Documentation: `RUN_REPORT.md`, `TUNING_GUIDE.md`, and `CODE_OVERVIEW.md` added to explain run steps, parameter tuning, and code structure.
- No algorithmic changes were made to `BPR.py`, `main_efficient.py`, or `utils.py` in this session; the new docs and small utility scripts reference and use existing CLI options.

The original README content follows below.

---

## Getting Started

### Prerequisites

```
Python 3.7
PyTorch >1.0
```


## How to Run 

Synthetic linear data experiments. Please download the dataset at

https://drive.google.com/file/d/1O52SlAHPRw_iFW_sAfm_vR3oMnoEb8am/view?usp=sharing

For the synthetic nonlinear data experiments, codes will be shared at

DAG-GNN https://github.com/fishmoon1234/DAG-GNN

### Synthetic linear data Experiments

CHOICE = nocurl, corresponding to the linear experiments, NoCurl-2 case in the paper

CHOICE = notear, corresponding to the linear experiments, NOTEARS case in the paper

LAMBDA1 = 10, corresponding to the parameter lambda_1 in the paper.

LAMBDA2 = 1000, corresponding to the parameter lambda_2 in the paper.


```
python main_efficient.py --data_variable_size=10 --graph_type="erdos-renyi" --repeat=100 --methods=<CHOICE> --h_tol=1e-8 --graph_degree=4 --lambda1=<LAMBDA1> --lambda2=<LAMBDA2> --data_type="synthetic"

```


## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{yu2021dag,
  title={DAGs with No Curl: An Efficient DAG Structure Learning Approach},
  author={Yue Yu, Tian Gao, Naiyu Yin and Qiang Ji},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
```


## Acknowledgments
Our work and code benefit from two existing works, which we are very grateful.

* DAG NOTEAR https://github.com/xunzheng/notears
* DAG NOFEAR https://github.com/skypea/DAG_No_Fear

