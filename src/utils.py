import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -- Paths ----------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_PROC  = ROOT_DIR / "data" / "processed"
REPORTS    = ROOT_DIR / "reports" / "figures"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlflow")

# -- Logging --------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
