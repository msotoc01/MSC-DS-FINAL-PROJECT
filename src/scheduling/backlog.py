"""
Building a backlog: the set of tasks already waiting at the start of a planning
period.

This lives in src/ and not in a script because it defines the evaluation
scenario, and three different entry points depend on it: the smoke test, the
experiments and the interface. A single definition means the three cannot drift
apart and report results from subtly different scenarios.
"""

import copy
import random
import config


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
    window = int(capacity * config.WINDOW_SPAN_HORIZONS)          # arrivals accumulate over one horizon's worth of time

    arrivals = sorted(t.arrival_time for t in tasks)

    # Viable windows
    viable = []
    for start in arrivals:
        inside = [t for t in tasks if start <= t.arrival_time < start + window]
        projects = {t.project_id for t in inside}
        work = sum(t.estimated_duration for t in inside)
        if len(projects) >= config.PROJECTS_PER_BACKLOG and work >= target_work * config.WINDOW_WORK_MARGIN:
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
    chosen = set(rng.sample(all_projects, config.PROJECTS_PER_BACKLOG))

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