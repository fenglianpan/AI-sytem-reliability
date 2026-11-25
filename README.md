# AI-sytem-reliability
This repository contains the simulation study code for evaluating model robustness under misspecification for a multi-stage AI reliability model that accounts for error propagation between upstream and downstream modules.
**Requirements:** pip install numpy scipy pandas matplotlib
**How to Run:** python misspecification_fast_clem.py
**Before running**, update the two output paths near the bottom of the script to match your machine:
    - df_rep.to_csv('/your/path/misspecification_per_rep_ae.csv', index=False)
    - plt.savefig('/your/path/misspecification_MAE_lineplot.pdf', index=False)
**Output: ** After running, two files are saved to the path defined in the script
    - misspecification_per_rep_ae.csv
    - misspecification_MAE_lineplot.pdf
**Reproducibility:** The script uses fixed random seeds to ensure identical results across runs:
    - np.random.seed(2024) — dataset generation
    - np.random.seed(42) — benchmark model fitting
    - np.random.seed(0) — CLEM, Musa-Okumoto, and Gompertz restart initialization
**Contact:** Fenglian Pan · fpan4@charlotte.edu · University of North Carolina at Charlotte
