import multiprocessing
from src.training.experiment_runner import main as run_experiments

if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for num_workers > 0 on Windows
    run_experiments()
