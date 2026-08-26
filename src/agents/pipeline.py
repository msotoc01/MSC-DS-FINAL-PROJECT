"""
Orchestrates the agent pipeline: retrieval -> inference -> complete Tasks.

This is what the Streamlit app and the experiment scripts call. It takes tasks
whose inferable attributes are None and returns the same tasks with those
attributes filled in.
"""

import time
import pandas as pd

from agents import inference_agent, retrieval_agent


def enrich_tasks(tasks, historical_df: pd.DataFrame,
                 embeddings=None, log_path=None, verbose: bool = True):
    """Fill in the inferred attributes of every task.

    Returns (tasks, log).
    """
    if embeddings is None:
        embeddings = retrieval_agent.get_index(historical_df)

    # All descriptions encoded in one pass: several times faster than looping,
    # because the encoder batches them onto the same tensor.
    descriptions = [t.description for t in tasks]
    neighbours = retrieval_agent.retrieve(descriptions, historical_df, embeddings)

    rows = []
    started = time.perf_counter()

    for i, (task, examples) in enumerate(zip(tasks, neighbours), start=1):
        result = inference_agent.infer_attributes(task.description, examples)

        # Only the fields the agents are responsible for.
        task.category = result["category"]
        task.context_group = result["context_group"]
        task.estimated_duration = result["estimated_duration"]
        task.priority = result["priority"]
        task.priority_level = result["priority_level"]

        rows.append({
            "task_id": task.task_id,
            "description": task.description,
            "category": result["category"],
            "estimated_duration": result["estimated_duration"],
            "priority": result["priority"],
            "repaired": "|".join(result["repaired"]),
            "top_similarity": round(float(examples["similarity"].iloc[0]), 4),
            "neighbour_categories": "|".join(examples["category"]),
        })

        if verbose and (i % 25 == 0 or i == len(tasks)):
            print(f"  inferred {i}/{len(tasks)}", end="\r")

    log = pd.DataFrame(rows)

    if verbose:
        print(f"  inferred {len(tasks)}/{len(tasks)} in {time.perf_counter() - started:.1f}s")
        print(_summary(log))

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)

    return tasks, log


def _summary(log: pd.DataFrame) -> str:
    """One block reporting how much of the batch the model actually answered.
    """
    repaired = log["repaired"].fillna("")
    clean = (repaired == "").sum()
    total_fallback = (repaired.str.count(r"\|") == 2).sum()

    lines = [f"  model answered cleanly: {clean}/{len(log)} ({clean / len(log):.0%})"]
    for field in ("category", "duration", "priority"):
        n = repaired.str.contains(field).sum()
        if n:
            lines.append(f"  repaired {field}: {n}")
    if total_fallback:
        lines.append(f"  WARNING: {total_fallback} tasks fell back entirely "
                     f"(the model gave nothing usable)")
    return "\n".join(lines)


def enrich_from_descriptions(descriptions, historical_df, embeddings=None):
    """Infer attributes for free-text descriptions typed by a user.
    """
    if embeddings is None:
        embeddings = retrieval_agent.get_index(historical_df)
    if isinstance(descriptions, str):
        descriptions = [descriptions]

    neighbours = retrieval_agent.retrieve(descriptions, historical_df, embeddings)
    return [
        {**inference_agent.infer_attributes(text, examples), "examples": examples}
        for text, examples in zip(descriptions, neighbours)
    ]