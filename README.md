# AI Reliability with Error Propagation

This repository contains the simulation study code for evaluating model robustness under misspecification for a multi-stage AI reliability model that accounts for error propagation between upstream and downstream modules.

## Requirements

```sh
pip install numpy scipy pandas matplotlib
```

## How to Run

```sh
python misspecification_fast_clem.py
```

Before running, update the two output paths near the bottom of the script to match your machine:

```py
df_rep.to_csv('/your/path/misspecification_per_rep_ae.csv', index=False)
plt.savefig('/your/path/misspecification_MAE_lineplot.pdf', bbox_inches='tight')
```

## Output

After running, two files are saved to the path defined in the script:

- `misspecification_per_rep_ae.csv` — absolute error per replication, method, and scenario  
- `misspecification_MAE_lineplot.pdf` — mean MAE line plot comparing all methods across scenarios

## Reproducibility

The script uses fixed random seeds to ensure identical results across runs:

- `np.random.seed(2024)` — dataset generation  
- `np.random.seed(42)` — benchmark model fitting  
- `np.random.seed(0)` — CLEM, Musa-Okumoto, and Gompertz restart initialization

## Contact

Fenglian Pan · [fpan4@charlotte.edu](mailto:fpan4@charlotte.edu) · University of North Carolina at Charlotte