from pathlib import Path


# -------------------------------------- important file names --------------------------------------
LOGS_FILE_NAME = "training.log"
CONFIG_TEMPLATE_NAME = "base_config"

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

