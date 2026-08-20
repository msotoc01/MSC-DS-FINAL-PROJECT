"""
Smoke test: load real tasks, build a batch, run every baseline, compare.

This is the end-to-end check of the scheduling half of the system:
    CSV -> Task -> ordering -> timed schedule -> metrics

It deliberately uses GROUND-TRUTH attributes rather than inferred ones, so any
difference between methods comes from the scheduling decision alone.
"""

import argparse
import sys
import copy
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          
sys.path.insert(0, str(ROOT / "src"))

from dataset.loaders import load_evaluation_tasks 
from scheduling.baselines import ALL_BASELINES    
from scheduling.schedule import check_dependencies, as_working_days
from evaluation import metrics   
from scheduling import milp  
import config            

# The chosen window must hold more work than the backlog needs
WINDOW_WORK_MARGIN = 1.5
# Acumulated work of 3 weks
WINDOW_SPAN_DAYS = 3


def build_batch(tasks, work_days=None, seed=42):
    """Build a BACKLOG: the set of tasks already waiting at the start of a
    planning period.

    A batch is not a stream of arrivals — it is what sits in the queue on Monday
    morning. Continuous arrival is modelled separately, through the dynamic
    re-optimisation mechanism (proposal §4.3).
    """
    rng = random.Random(seed)

    if work_days is None:
        work_days = config.WORK_DAYS_PER_BACKLOG

    capacity = work_days * config.WORKING_MINUTES_PER_DAY
    target_work = capacity * config.OVERLOAD_FACTOR
    window = int(capacity * WINDOW_SPAN_DAYS)          # arrivals accumulate over one horizon's worth of time

    arrivals = sorted(t.arrival_time for t in tasks)

    # Viable windows
    viable = []
    for start in arrivals:
        inside = [t for t in tasks if start <= t.arrival_time < start + window]
        projects = {t.project_id for t in inside}
        work = sum(t.estimated_duration for t in inside)
        if len(projects) >= config.MAX_PROJECTS_PER_BACKLOG and work >= target_work * WINDOW_WORK_MARGIN:
            viable.append(start)

    if not viable:
        raise ValueError(
            f"No window holds enough work for {work_days} working days at "
            f"overload {config.OVERLOAD_FACTOR}. Reduce WORK_DAYS_PER_BACKLOG "
            f"or OVERLOAD_FACTOR in config.py."
        )

    # Pick a window, then the projects worked on during it
    start = rng.choice(viable)
    inside = [t for t in tasks if start <= t.arrival_time < start + window]

    all_projects = sorted({t.project_id for t in inside})     # sorted => reproducible
    chosen = set(rng.sample(all_projects, config.MAX_PROJECTS_PER_BACKLOG))

    candidates = sorted(
        (t for t in inside if t.project_id in chosen),
        key=lambda t: t.arrival_time,        # oldest queued first
    )

    # Fill the backlog up to the overloaded target
    batch, work = [], 0
    for t in candidates:
        if work >= target_work:
            break
        batch.append(t)
        work += t.estimated_duration

    batch = copy.deepcopy(batch)      # never mutate the caller's tasks

    # 6. Everything is already queued at t=0
    for t in batch:
        t.deadline = t.deadline - t.arrival_time   # slack, measured from t = 0
        t.arrival_time = 0            # no arrival constraint inside a backlog

    return batch


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

    # Show one schedule in full
    # sample_method = "EDF"
    # print(f"\n-- Schedule produced by {sample_method} (first 10 rows) --")
    # print(schedules[sample_method].head(10).to_string(index=False))


if __name__ == "__main__":
    main()