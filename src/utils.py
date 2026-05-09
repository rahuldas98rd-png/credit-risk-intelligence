import logging
import os
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
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
