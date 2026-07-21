"""Project-level configuration values."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "figures"

TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_SELECTION_METRIC = "F1 Score"