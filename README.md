# Hyperparameter Optimization for LSTM-Based Electricity Load Forecasting

A comparison of hyperparameter optimization methods for a global LSTM forecasting model trained on the [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset.

Four methods are compared under an equal function-evaluation budget:

| Method | Description |
|--------|-------------|
| Baseline | Fixed default hyperparameters (hidden\_dim=128, lr=0.001, dropout=0.0) |
| Random Search | Uniform random sampling over the hyperparameter space |
| PSO | Particle Swarm Optimization (single-objective, minimizes validation MSE) |
| MOO | Multi-objective optimizer (NSGA-II variant); minimizes validation MSE and model complexity jointly |

---

## Dataset

**ElectricityLoadDiagrams20112014** — 370 Portuguese households, 15-minute resolution, 2011–2014.

- Source: UCI ML Repository
- Raw file: `data/raw/LD2011_2014.txt` (not tracked by git — download separately)
- Preprocessing: leading-zero imputation → hourly aggregation → chronological split → 100-household selection → per-household min-max normalization

| Split | Period | Hours |
|-------|--------|-------|
| Train | 2011-01-01 – 2013-12-31 | ~26,280 |
| Val   | 2014-01-01 – 2014-06-30 | ~4,344  |
| Test  | 2014-07-01 – 2014-12-31 | ~4,416  |

---

## Model

**Global LSTM** — a single model trained across all 100 households simultaneously.

- Input: 24-hour window of normalized load values `(batch, 24, 1)`
- Output: next 24-hour forecast `(batch, 24)`
- Hyperparameters searched: `hidden_dim` ∈ [32, 256], `lr` ∈ [1e-4, 5e-3], `dropout` ∈ [0.0, 0.3]
- Training uses cosine annealing LR schedule and early stopping

---

## Project Structure

```
MOO-Electricity-Forecast/
├── data/
│   ├── raw/                        # raw dataset (not tracked)
│   └── processed/                  # preprocessed CSVs and scaling JSON
├── checkpoints/                    # saved model weights
├── results/                        # JSON metrics from each method
├── src/
│   ├── config.py                   # all hyperparameters and mode-specific settings
│   ├── models/
│   │   └── lstm.py                 # LSTM model definition
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset (sliding window)
│   │   ├── preprocess.py           # preprocessing functions
│   │   └── run_preprocessing.py    # preprocessing entry point
│   ├── optimizers/
│   │   ├── pso.py                  # Particle Swarm Optimization
│   │   └── moo.py                  # Multi-objective optimizer (NSGA-II)
│   ├── training/
│   │   ├── trainer.py              # train_one_epoch / validate
│   │   ├── training_pipeline.py    # train_single_configuration / retrain_and_evaluate
│   │   ├── fitness.py              # PSO and MOO fitness functions
│   │   ├── early_stopping.py       # early stopping with checkpoint saving
│   │   └── experiment_runner.py    # main experiment loop
│   └── utils/
│       └── seed.py                 # reproducibility seed setter
└── main.py                         # entry point
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download the raw dataset from UCI and place it at `data/raw/LD2011_2014.txt`.

---

## Preprocessing

Run once to generate the processed data files:

```bash
python -m src.data.run_preprocessing
```

This produces:
- `data/processed/electricity_train.csv`
- `data/processed/electricity_val.csv`
- `data/processed/electricity_test.csv`
- `data/processed/electricity_scaling.json`
- `data/processed/selected_households.json`

---

## Running Experiments

```bash
python main.py
```

By default runs in `mode="full"` (100 households, 50 epochs, full optimizer budgets). Edit `src/training/experiment_runner.py` to enable/disable individual methods.

**Dev mode** (10 households, 2 epochs, minimal budgets — for quick testing):

```python
# In experiment_runner.py main():
config = Config(mode="dev")
```

### Mode comparison

| Setting | Dev | Full |
|---------|-----|------|
| Households | 10 | 100 |
| Batch size | 512 | 2048 |
| Search epochs (HPO) | 10 (+ early stop, patience 3) | 20 (+ early stop, patience 5) |
| Retrain epochs (final) | 15 | 60 |
| LR | 0.001 | 0.004 |
| DataLoader workers | 0 | 4 |
| Eval budget (all methods) | 12 | 30 |
| Random trials | 12 | 30 |
| PSO swarm / iterations | 4 / 2 | 6 / 4 |
| MOO pop / generations | 4 / 2 | 6 / 4 |

---

## Reproducibility

All experiments seed Python, NumPy, and PyTorch via `set_seed(42)`. CuDNN deterministic mode is enabled in `seed.py`; `benchmark=True` is re-enabled in `experiment_runner.py` for throughput.
