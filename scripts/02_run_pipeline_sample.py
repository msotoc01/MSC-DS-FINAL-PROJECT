"""
Smoke test: load real tasks, build a batch, run every baseline, compare.

This is the end-to-end check of the scheduling half of the system:
    CSV -> Task -> ordering -> timed schedule -> metrics

It deliberately uses GROUND-TRUTH attributes rather than inferred ones, so any
difference between methods comes from the scheduling decision alone. Once the
agent pipeline exists this becomes the "perfect information" arm of the
comparison required by proposal §6, and a second arm is added using
pipeline.enrich_tasks().

"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          
sys.path.insert(0, str(ROOT / "src"))

from dataset.loaders import load_evaluation_tasks 
from scheduling.baselines import ALL_BASELINES    
from scheduling.schedule import check_dependencies, as_working_days
from evaluation import metrics                 

MINUTES_PER_DAY = 24 * 60


def build_batch(tasks, window_days=1, seed=42):
    """Take every task arriving inside one time window, then reset the clock.

    Why a window and not a random sample: arrivals are spread over ~58 days, so
    random tasks never compete for time — the executor simply waits for each one
    and every method returns the same schedule. A window gives a realistic burst
    of work where tasks contend and ordering actually matters.

    The clock is shifted so the earliest arrival sits at t=0, which keeps
    makespan comparable across batches.
    """
    window = window_days * MINUTES_PER_DAY
    arrivals = sorted(t.arrival_time for t in tasks)

    # Pick the busiest window so the batch is genuinely contended.
    best_start, best_count = arrivals[0], 0
    for start in arrivals:
        count = sum(1 for a in arrivals if start <= a < start + window)
        if count > best_count:
            best_start, best_count = start, count

    batch = [t for t in tasks if best_start <= t.arrival_time < best_start + window]

    # Shift the clock: earliest arrival becomes t=0.
    offset = min(t.arrival_time for t in batch)
    for t in batch:
        t.arrival_time -= offset
        t.deadline -= offset

    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = load_evaluation_tasks(with_ground_truth=True)
    batch = build_batch(tasks, window_days=args.window_days, seed=args.seed)

    total_work = sum(t.estimated_duration for t in batch)
    print("=" * 72)
    print(f"BATCH: {len(batch)} tasks | total work {total_work} min ")


if __name__ == "__main__":
    main()