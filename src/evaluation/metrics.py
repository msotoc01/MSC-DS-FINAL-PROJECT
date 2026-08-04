"""
All metrics in one place — the two-level evaluation framework (proposal §6).

Nivel 1 (attribute inference): accuracy Y F1 para los TRES atributos
categóricos (category, context_group, priority) + MAE para duración.

Nivel 2 (scheduling quality): las TRES métricas del proposal — makespan,
nº de context switches y deadline satisfaction rate. (Total lateness puede
calcularse como métrica auxiliar de diagnóstico, pero la comparación
principal del report usa las tres del proposal.)
"""

import pandas as pd


def makespan(schedule: pd.DataFrame) -> float:

    if len(schedule) == 0: return 0

    start = schedule["start_min"].min()
    end = schedule["end_min"].max()
    timespan = end - start

    return timespan


def context_switches(schedule: pd.DataFrame) -> int:
    """Number of times the context_group changes between consecutive tasks.
    The first task never counts as a switch (it has no predecessor)."""

    if len(schedule) == 0:
        return 0

    ordered = schedule.sort_values("position")
    changes = ordered["context_group"] != ordered["context_group"].shift()
    changes = changes.iloc[1:]

    return changes.sum()


def deadline_satisfaction(schedule: pd.DataFrame) -> float:
    """Fraction of tasks finishing on time (0-1). NaN for an empty schedule so it
    is excluded from averages across batches."""

    if len(schedule) == 0: return float("nan")

    satisfaction = schedule["end_min"] <= schedule["deadline"]
    mean_stisfaction = satisfaction.mean()
    
    return mean_stisfaction


def total_lateness(schedule: pd.DataFrame) -> float:

    if len(schedule) == 0: return 0

    difference = schedule["end_min"] - schedule["deadline"]
    lateness = difference.clip(lower=0)

    return lateness.sum()


def scheduling_summary(schedules: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Method x metric table comparing every scheduler on the same batch.

    This is the central results table of the report and of the comparison tab
    in the Streamlit app. Values are returned unrounded: rounding is a
    presentation concern, and these numbers get averaged across batches in
    scheduling_eval, where premature rounding would lose precision.
    """
    rows = []
    for name, sched in schedules.items():
        rows.append({
            "method": name,
            "makespan_min": makespan(sched),
            "context_switches": context_switches(sched),
            "deadline_satisfaction": deadline_satisfaction(sched),
            "total_lateness": total_lateness(sched),
        })
    return pd.DataFrame(rows).set_index("method")
