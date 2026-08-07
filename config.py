"""
Global configuration: paths, model identifiers and optimisation weights.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # where scripts/01_generate_dataset.py writes

HISTORICAL_CSV = RAW_DIR / "historical_repository.csv"
EVALUATION_CSV = RAW_DIR / "evaluation_set.csv"              # visible columns only
EVALUATION_GT_CSV = RAW_DIR / "evaluation_ground_truth.csv"  # hidden attributes

# Reproducibility
RANDOM_SEED = 42        # default seed for dataset generation and batching

# Dataset sizes actually generated (see scripts/01_generate_dataset.py args).
N_HISTORICAL_TASKS = 4500
N_EVALUATION_TASKS = 600

# MILP / scheduling
W_MAKESPAN = 1.0
W_CONTEXT_SWITCH = 1.0
W_DEADLINE_VIOLATION = 1.0

W_PROJECT_SWITCH = 0.0

CONTEXT_SWITCH_COST_MIN = 5    # minutes lost when consecutive tasks differ in context_group
SOLVER_TIME_LIMIT_S = 60       # CBC time limit; accept best feasible solution beyond it

WORKING_MINUTES_PER_DAY = 480

LLM_MODEL_ID = "gpt-4o-mini"