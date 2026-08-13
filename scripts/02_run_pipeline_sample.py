"""
Smoke test: load real tasks, build a batch, run every baseline, compare.

This is the end-to-end check of the scheduling half of the system:
    CSV -> Task -> ordering -> timed schedule -> metrics

It deliberately uses GROUND-TRUTH attributes rather than inferred ones, so any
difference between methods comes from the scheduling decision alone.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          
sys.path.insert(0, str(ROOT / "src"))

from dataset.loaders import load_evaluation_tasks 
from scheduling.baselines import ALL_BASELINES    
from scheduling.schedule import check_dependencies, as_working_days
from evaluation import metrics   
from scheduling import milp              

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
    print(f"BATCH: {len(batch)} tasks | total work {total_work} min "
          f"({total_work / 60:.1f} h) | window {args.window_days} day(s)")
    print(f"BATCH: {len(batch)} tasks | total work {as_working_days(total_work)}")
    print(f"context groups present: {sorted({t.context_group for t in batch})}")
    print(f"projects present: {len({t.project_id for t in batch})}")
    print("=" * 72)

    # Run every baseline on the same batch
    schedules = {}
    for name, scheduler in ALL_BASELINES.items():
        if scheduler is None:          # Agent-only not implemented yet
            continue
        schedules[name] = scheduler(list(batch))

    # Compare
    print("\n-- Metrics by method --")
    print(metrics.scheduling_summary(schedules).round(3).to_string())

    #  MILP
    sched, info = milp.solve(list(batch))
    print(f"\nMILP: {info['status']} in {info['solve_time_s']}s "
        f"(objective {info['objective']:.0f})")
    if len(sched):
        schedules["MILP"] = sched

    # Dependency violations (baselines do not enforce them)
    print("\n-- Dependency violations per method --")
    for name, sched in schedules.items():
        order = {t.task_id: t for t in batch}
        ordered_tasks = [order[tid] for tid in sched.sort_values("position")["task_id"]]
        print(f"  {name:10s} {len(check_dependencies(ordered_tasks))}")

    # Show one schedule in full
    # sample_method = "EDF"
    # print(f"\n-- Schedule produced by {sample_method} (first 10 rows) --")
    # print(schedules[sample_method].head(10).to_string(index=False))


if __name__ == "__main__":
    main()