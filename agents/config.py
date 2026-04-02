"""Shared configuration for monitoring agents."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
CHALLENGER_DIR = MODELS_DIR / "challenger"

# Supabase
SUPABASE_DB_URL = os.environ.get("DATABASE_URL", "")

# Drift thresholds
PSI_WARNING = 0.10   # score distribution drift warning
PSI_CRITICAL = 0.25  # score distribution drift → retrain
CSI_THRESHOLD = 0.20 # per-feature characteristic stability index

# Performance thresholds
AUC_DROP_THRESHOLD = 0.03  # 3-point AUC drop from training → retrain

# Decision thresholds (mirrored from API)
APPROVE_THRESHOLD = 0.15
REVIEW_THRESHOLD = 0.30
