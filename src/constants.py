from pathlib import Path


# -------------------------------------- important file / folder names --------------------------------------
LOGS_FILE_NAME = "training.log"
CHECKPOINTS_FOLDER_NAME = "checkpoints"
BEST_CHECKPOINT_NAME = "best_checkpoint.pth"
PLOTS_FOLDER_NAME = "plots"

# -------------------------------------- Paths realtive to project root --------------------------------------
CONFIGS_PATH = Path("configs")
DATA_DIR = Path("test_cases/public/instances")
PRODUCTION_MODELS = Path("production_models")
HYDRA_OUTPUT = Path("results")


SPLIT_TO_PATH = {
    "training": DATA_DIR / "train_processed.json",
    "validation": DATA_DIR / "val_processed.json",
    "testing": DATA_DIR / "test_processed.json",
}

# -------------------------------------- Some other constants --------------------------------------
MAX_AGE = 102
CONFIG_TEMPLATE_NAME = "base_config"