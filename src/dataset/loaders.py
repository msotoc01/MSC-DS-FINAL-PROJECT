"""
Loading the generated CSVs back into Task objects.

Kept out of the scripts because several of them need it: the smoke test, the
experiments, the Streamlit app and the retrieval agent all start here.

Note on the evaluation set: it is deliberately split in two files. Only
evaluation_set.csv is visible to the agent pipeline; the ground truth is loaded
separately and used either to score the agents, or — as in the scheduler smoke
test — to run the schedulers on perfect attributes and isolate scheduling error
from inference error.
"""

import pandas as pd

import config
from src.task import Task


def _parse_dependencies(value) -> list[str]:
    """The CSV stores dependencies as 'id1|id2', or empty when there are none.
    pandas reads those empty cells as NaN, so handle that before splitting:
    ''.split('|') would give [''], a list containing an empty string."""
    if pd.isna(value) or value == "":
        return []
    return str(value).split("|")


def _rows_to_tasks(df: pd.DataFrame) -> list[Task]:
    """Build Task objects from a DataFrame whose columns match the dataclass
    fields exactly (that is why generator and Task share one definition)."""
    tasks = []
    for row in df.to_dict("records"):
        row["dependencies"] = _parse_dependencies(row["dependencies"])
        tasks.append(Task(**row))
    return tasks


def load_historical_repository() -> list[Task]:
    """The full knowledge base: 4,500 tasks with all attributes known."""
    df = pd.read_csv(config.HISTORICAL_CSV)
    return _rows_to_tasks(df)


def load_evaluation_tasks(with_ground_truth: bool = True) -> list[Task]:
    """Evaluation tasks.

    with_ground_truth=True  -> complete Tasks (for testing schedulers alone)
    with_ground_truth=False -> only description/arrival/deadline are real;
                               the attributes the agents must infer are left
                               as None, which is what the pipeline receives.
    """
    visible = pd.read_csv(config.EVALUATION_CSV)

    if with_ground_truth:
        truth = pd.read_csv(config.EVALUATION_GT_CSV)
        df = pd.merge(visible, truth, on="task_id", validate="one_to_one")
        return _rows_to_tasks(df)

    return [
        Task(task_id=r["task_id"], project_id=None, description=r["description"],
             domain=None, category=None, context_group=None,
             estimated_duration=None, priority=None, priority_level=None,
             arrival_time=r["arrival_time"], deadline=r["deadline"],
             variation_level=None, dependencies=[])
        for r in visible.to_dict("records")
    ]