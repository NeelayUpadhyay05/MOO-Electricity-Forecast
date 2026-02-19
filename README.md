# MOO for Electricity Load Forecasting

Musk Ox Optimizer (MOO) metaheuristic for tuning GRU/LSTM hyperparameters on UCI ElectricityLoadDiagrams2011–2014 dataset (Portuguese clients, short-term hourly forecasting).

[![status](https://img.shields.io/badge/status-WIP-yellow.svg)](https://github.com/NeelayUpadhyay05/MOO-Electricity-Forecast)

## Features
- Sliding window TS setup (LL=24 → 1-step ahead).
- MOO vs random/grid baselines.
- Metrics: RMSE/MAE on test.

## Setup
### Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate moo-electricity
```

### Pip

```bash
pip install -r requirements.txt
```

## Usage

```bash
conda activate moo-electricity
code .
```
Open `notebooks/01_data_prep_baselines.ipynb`.

## Structure

```text
├── data/ # UCI raw/processed
├── notebooks/ # Analysis (numbered)
├── models/ # .pth saves (gitignore)
├── environment.yml # Repro
└── requirements.txt
```


## License
MIT [LICENSE](LICENSE)
****
