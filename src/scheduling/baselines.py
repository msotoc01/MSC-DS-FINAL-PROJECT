"""
The 5 baseline schedulers.

All of them decide only the ORDER; times are assigned by
sequence_to_schedule(), identically for every method, so the comparison is fair.

Design decision: baselines do NOT enforce dependencies. They are simple
dispatch rules — adding dependency logic would turn them into a weaker version
of the proposed system and blur the comparison. Violations are measured with
schedule.check_dependencies() and reported as a result in their own right.
"""

from scheduling.schedule import sequence_to_schedule
from task import Task


def fifo(tasks: list[Task]):
    """First in, first out: earliest arrival first."""
    ordered = sorted(tasks, key=lambda t: t.arrival_time)
    return sequence_to_schedule(ordered)


def shortest_task_first(tasks: list[Task]):
    """Shortest estimated duration first."""
    ordered = sorted(tasks, key=lambda t: t.estimated_duration)
    return sequence_to_schedule(ordered)


def earliest_deadline_first(tasks: list[Task]):
    """Earliest deadline first."""
    ordered = sorted(tasks, key=lambda t: t.deadline)
    return sequence_to_schedule(ordered)


def priority_first(tasks: list[Task]):
    """Highest priority first; ties broken by earliest deadline."""
    ordered = sorted(tasks, key=lambda t: (-t.priority_level, t.deadline))
    return sequence_to_schedule(ordered)


ALL_BASELINES = {
    "FIFO": fifo,
    "STF": shortest_task_first,
    "EDF": earliest_deadline_first,
    "Priority": priority_first,
}
