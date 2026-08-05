"""
Common output format for ALL schedulers (MILP and baselines).

A "schedule" is a DataFrame with one row per task and fixed columns. Because
every scheduler returns this exact format, metrics and plots are written once
and work for all of them.

Design decisions (documented in the report):
  - Single executor, tasks run in series (proposal's positional model).
  - A task cannot start before it arrives:  start = max(clock, arrival_time).
  - Switching cost is paid BETWEEN tasks, pushing the next start forward, and
    therefore counts towards makespan. It is charged only when two tasks of
    different context_group run back-to-back WITHOUT an idle gap in between
    (if the executor already waited for arrival, no extra cognitive cost).
  - The first task never pays a switch (no previous context).
  - This function only computes times. It does NOT validate dependencies;
    use check_dependencies() for that.
"""

import pandas as pd

from src.task import Task
import config

SCHEDULE_COLUMNS = [
    "task_id", "position", "start_min", "end_min",
    "context_group", "priority_level", "deadline", "estimated_duration",
    "switch_before",
]


def sequence_to_schedule(ordered_tasks: list[Task],
                         switch_cost: int = config.CONTEXT_SWITCH_COST_MIN) -> pd.DataFrame:
    """Turn an ORDERED list of tasks into a timed schedule."""
    rows = []
    clock = 0
    prev_context = None

    for position, t in enumerate(ordered_tasks):
        earliest = max(clock, t.arrival_time)

        switch = 0
        if prev_context is not None and t.context_group != prev_context and earliest == clock:
            switch = switch_cost

        start = earliest + switch
        end = start + t.estimated_duration

        rows.append({
            "task_id": t.task_id,
            "position": position,
            "start_min": start,
            "end_min": end,
            "context_group": t.context_group,
            "priority_level": t.priority_level,
            "deadline": t.deadline,
            "estimated_duration": t.estimated_duration,
            "switch_before": switch,
        })

        clock = end
        prev_context = t.context_group

    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def check_dependencies(ordered_tasks: list[Task]) -> list[tuple]:
    """Return (task_id, missing_dependency_id) pairs where a task precedes a
    dependency it needs. Empty list => order is legal."""
    position = {t.task_id: i for i, t in enumerate(ordered_tasks)}
    violations = []
    for t in ordered_tasks:
        for dep in t.dependencies:
            if dep in position and position[dep] > position[t.task_id]:
                violations.append((t.task_id, dep))
    return violations

def as_working_days(minutes: int) -> str:
    """Format a duration in working minutes as 'Xd Yh' for display."""
    days, rem = divmod(int(minutes), config.WORKING_MINUTES_PER_DAY)
    hours = rem / 60
    return f"{days}d {hours:.1f}h" if days else f"{hours:.1f}h"