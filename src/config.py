class Config:
    def __init__(self, mode="dev"):

        # -----------------------
        # Mode
        # -----------------------
        self.mode = mode

        # -----------------------
        # Dev Controls
        # -----------------------
        self.dev_households = 10
        self.dev_timesteps = 2000

        # -----------------------
        # Model
        # -----------------------
        self.hidden_dim = 64
        self.dropout = 0.2

        # -----------------------
        # Training (mode-specific)
        # -----------------------
        if mode == "full":
            self.lr = 0.004          # linear LR scaling: 0.001 × (2048/512)
            self.batch_size = 2048
            self.epochs = 50
            self.patience = 10
            self.num_workers = 4
        else:
            self.lr = 0.001
            self.batch_size = 512
            self.epochs = 15
            self.patience = 5
            self.num_workers = 0

        self.min_delta = 1e-4
        self.checkpoint_path = "checkpoints/temp_best.pt"

        # -----------------------
        # DataLoader
        # -----------------------
        self.pin_memory = True                            # no-op on CPU; speeds up CPU→GPU on CUDA
        self.drop_last = True                             # avoid noisy final batch during training
        self.persistent_workers = (self.num_workers > 0) # keep workers alive between epochs

        # -----------------------
        # Search Budgets (mode-specific)
        # -----------------------
        if mode == "full":
            self.random_trials   = 20
            self.pso_swarm_size  = 10
            self.pso_iterations  = 10
            self.moo_pop_size    = 20
            self.moo_generations = 15
        else:
            self.random_trials   = 6
            self.pso_swarm_size  = 4
            self.pso_iterations  = 2
            self.moo_pop_size    = 6
            self.moo_generations = 2

        # -----------------------
        # Hyperparameter Bounds (single source of truth)
        # -----------------------
        self.hp_bounds = {
            "hidden_dim": [32, 256],
            "lr":         [1e-4, 5e-3],
            "dropout":    [0.0, 0.3],
        }
