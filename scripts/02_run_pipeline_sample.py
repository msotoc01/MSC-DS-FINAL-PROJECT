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
from scheduling.backlog import build_batch
from evaluation import metrics   
from scheduling import milp  
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-days", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = load_evaluation_tasks(with_ground_truth=True)
    batch = build_batch(tasks, work_days=args.work_days, seed=args.seed)

    work_days = args.work_days or config.WORK_DAYS_PER_BACKLOG
    capacity = work_days * config.WORKING_MINUTES_PER_DAY
    total_work = sum(t.estimated_duration for t in batch)

    print("=" * 72)
    print(f"BACKLOG: {len(batch)} tasks waiting | {as_working_days(total_work)} of work")
    print(f"  horizon capacity: {as_working_days(capacity)} " 
          f"-> overload {total_work / capacity:.2f}x")
    print(f"  projects: {len({t.project_id for t in batch})} | " 
          f"context groups: {sorted({t.context_group for t in batch})}")
    print("=" * 72)

    # Every scheduler runs on the same backlog
    schedules = {}
    for name, scheduler in ALL_BASELINES.items():
        if scheduler is None:          # Agent-only not implemented yet
            continue
        schedules[name] = scheduler(list(batch))

    milp_schedule, info = milp.solve(list(batch))
    print(f"\nMILP solver: {info['status']} in {info['solve_time_s']}s " 
          f"(objective {info['objective']:.0f})")
    if len(milp_schedule):
        schedules["MILP"] = milp_schedule

    # Compare
    print("\n-- Metrics by method --")
    print(metrics.scheduling_summary(schedules).round(3).to_string())

    # Dependency violations (baselines do not enforce them)
    print("\n-- Dependency violations per method --")
    lookup = {t.task_id: t for t in batch}
    for name, sched in schedules.items():
        ordered_tasks = [lookup[tid] for tid in sched.sort_values("position")["task_id"]]
        print(f"  {name:10s} {len(check_dependencies(ordered_tasks))}")


if __name__ == "__main__":
    main()