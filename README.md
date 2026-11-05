# A Data-Centric Validation Procedure for Detecting Non-Transferable Prognostics Models under Asset Scarcity

Code and reproduction instructions for the paper of the same name.

The work studies how a conventional machine-learning prognostics pipeline can look
accurate on a single held-out asset yet fail to transfer to new assets, and it
packages the diagnosis into four cheap checks: cross-asset (leave-one-unit-out)
validation, calibration analysis, SHAP-based explanation, and alert-timing.
The checks are shown to discriminate: they fail on the water-pump case and pass
on the C-MAPSS turbofan benchmark, which has genuine monotone degradation.

## Repository contents

| File | Purpose |
|------|---------|
| `explainable_rul_water_pump.py` | Main pipeline and all pump-case analysis: feature engineering, gradient-boosting ensemble, regime-aware isotonic calibration, Brier-score decomposition, leave-one-unit-out cross-validation, SHAP, cost-sensitive alerts, Weibull survival fit, and the figures/tables in the paper. |
| `protocol_cmapss.py` | Positive control. Applies the identical pipeline to C-MAPSS FD001 and runs the same four checks, to show they clear a model that genuinely transfers. |
| `requirements.txt` | Library versions used to produce the reported results. |

## Data

Both datasets are public and are **not** included in this repository.

- **Water pump run-to-failure dataset** (seven units): download from Kaggle
  and place the CSV where `explainable_rul_water_pump.py` expects it (see the
  path/config near the top of that script).
  Link: `<ADD KAGGLE URL>`
- **C-MAPSS turbofan degradation dataset** (FD001): download `train_FD001.txt`
  from the NASA Prognostics Center of Excellence data repository and place it in
  the working directory.
  Link: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

## Environment

Python 3.8 was used. To set up:

```bash
pip install -r requirements.txt
```

The reported numbers were produced with scikit-learn 1.3.2, numpy 1.22.4,
pandas 2.0.3, and shap 0.44.1. Results are deterministic given the fixed random
seeds set inside each script.

## Reproducing the results

### Pump case (main analysis)

```bash
python explainable_rul_water_pump.py
```

This runs the full pipeline on the pump dataset and writes the metrics, the
leave-one-unit-out results, the SHAP output, the alert analysis, and the figures
used in the paper. The headline result is the inverted test-unit performance
(AUC 0.200 at the 72-hour horizon) and the leave-one-unit-out spread from 0.08 to
0.99 across folds.

### C-MAPSS positive control

```bash
python protocol_cmapss.py --train_file train_FD001.txt \
    --n_units 20 --n_ensemble 10 --steps 1,2,3,4
```

This runs the four checks on C-MAPSS FD001 and writes per-fold CSVs plus a summary
table to `results_cmapss_fd001/`. All four checks pass, with leave-one-unit-out
AUC near 0.99.

Useful options:

- `--n_units 0` uses all 100 units (slower; the paper uses a 20-unit subset).
- `--steps 3,4` reruns only the SHAP and alert checks.
- `--cache_features` caches engineered features so reruns skip that step
  (requires `pyarrow`).

## Notes

- `protocol_cmapss.py` holds the pipeline fixed and changes only the dataset, so
  the pump-versus-turbofan comparison isolates the effect of the data rather than
  the model.
- Horizons for C-MAPSS are set in cycles (15, 30, 50) to match the shorter unit
  life, with a piece-wise remaining-life cap of 125 cycles.
