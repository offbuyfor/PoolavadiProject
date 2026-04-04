# PoolavadiProject

Small project for data preparation and sample model training.

## Contents

- [train_sample.py](train_sample.py) — training script.
- [topMoversDatPrep.ipynb](topMoversDatPrep.ipynb) — notebook for data prep.
- [sample_predictions.csv](sample_predictions.csv) — example data / predictions.

## Prerequisites

- Python 3.8+ recommended
- Git (optional)

## Setup

Windows PowerShell (from project root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # if present
```

If you don't have a `requirements.txt`, install common packages:

```powershell
pip install pandas numpy scikit-learn jupyter
```

## Run

- Run training script:

```powershell
python train_sample.py
```

- Open the notebook:

```powershell
jupyter notebook topMoversDatPrep.ipynb
```

## Data

Place or update `sample_predictions.csv` in the project root. The notebook and scripts read that file.

## Next steps

- Add a `requirements.txt` (I can generate one from the environment).
- Add usage examples or sample outputs.
- Deploy API + batch to GCP Cloud Run using [DEPLOY_GCP.md](DEPLOY_GCP.md).

If you'd like, I can generate `requirements.txt` now or run `train_sample.py` and report results.
