"""
All metrics in one place — the two-level evaluation framework.

Level 1, attribute inference: did the agents read the description correctly?
Compares the inferred attributes against the hidden ground truth.
 
Level 2, scheduling quality: was the resulting order a good one?
Measures the schedule the optimiser produced.
"""

import pandas as pd
from sklearn.metrics import classification_report, f1_score


# Level 2
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

# Level 1:

def duration_mae(inferred, actual) -> float:
    """Mean absolute error of the duration estimates, in minutes.
    """
    return (pd.Series(inferred).reset_index(drop=True)
            - pd.Series(actual).reset_index(drop=True)).abs().mean()
 
 
def duration_baselines(actual, categories, category_ranges) -> dict:
    """Reference points the MAE has to be read against.

    global_mean   predicting the same number for every task, ignoring text
    category_mid  the midpoint of the correct category
    """
    actual = pd.Series(actual).reset_index(drop=True)
    mid = pd.Series([sum(category_ranges[c]) / 2 for c in categories])
    return {
        "global_mean": (actual - actual.mean()).abs().mean(),
        "category_mid": (mid - actual).abs().mean(),
    }
 
 
def categorical_f1(inferred, actual) -> float:
    """Macro F1: every class weighs the same regardless of how common it is.
 
    Macro rather than weighted, because the categories are unbalanced and a
    model that got only the frequent ones right would still score well on a
    weighted average.
    """
    return f1_score(list(actual), list(inferred), average="macro", zero_division=0)
 
 
def inference_report(inferred, actual) -> pd.DataFrame:
    """Per-class precision, recall and F1 — where the confusions actually are."""
    report = classification_report(list(actual), list(inferred),
                                   output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(3)
 
 
def inference_summary(log: pd.DataFrame, truth: pd.DataFrame,
                      group_by: str = None) -> pd.DataFrame:
    """Level-1 metrics for a run, optionally broken down by a column.
 
    log needs columns: task_id, category, priority, estimated_duration.
    truth needs the same task_ids with the real values.
    """
    merged = log.merge(truth, on="task_id", suffixes=("_pred", "_true"))
    groups = ([(name, block) for name, block in merged.groupby(group_by)]
              if group_by else [("all", merged)])
 
    rows = []
    for name, block in groups:
        rows.append({
            group_by or "group": name,
            "n": len(block),
            "category_f1": categorical_f1(block.category_pred, block.category_true),
            "category_accuracy": (block.category_pred == block.category_true).mean(),
            "priority_f1": categorical_f1(block.priority_pred, block.priority_true),
            "duration_mae": duration_mae(block.estimated_duration_pred,
                                         block.estimated_duration_true),
        })
    return pd.DataFrame(rows).round(3)