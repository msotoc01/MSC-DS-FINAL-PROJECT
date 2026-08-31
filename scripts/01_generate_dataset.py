"""
Generate the synthetic dataset: historical repository + evaluation set (+ hidden
ground truth for the evaluation set).
"""

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dataset.generator import generate_projects, tasks_to_records 
import config

OUTPUT_DIR = config.RAW_DIR

EVAL_VISIBLE_COLUMNS = ["task_id", "description", "arrival_time", "deadline"]

EVAL_HIDDEN_COLUMNS = [
    "project_id",
    "domain",
    "category",
    "context_group",
    "estimated_duration",
    "priority",
    "priority_level",
    "variation_level",
    "dependencies",
    "template_source",
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Historical repository (full attributes, repo-only templates)
    repo_tasks = generate_projects(
        config.N_HISTORICAL_TASKS, random.Random(config.RANDOM_SEED),
        split="repo", id_prefix="task", project_prefix="proj"
    )
    
    repo_df = pd.DataFrame(tasks_to_records(repo_tasks))
    repo_df.to_csv(OUTPUT_DIR / "historical_repository.csv", index=False)

    # Evaluation set (mix of in-distribution + out-of-distribution templates)
    eval_tasks = generate_projects(
        config.N_EVALUATION_TASKS, random.Random(config.RANDOM_SEED + 1),
        split="eval", id_prefix="evaltask", project_prefix="evalproj"
    )
    eval_df = pd.DataFrame(tasks_to_records(eval_tasks))

    # Guard: every generated column must be explicitly classified as visible or hidden.
    # Without this, adding a field to Task would silently drop it from both CSVs.
    unassigned = set(eval_df.columns) - set(EVAL_VISIBLE_COLUMNS) - set(EVAL_HIDDEN_COLUMNS)
    assert not unassigned, f"Columns not assigned to visible/hidden: {sorted(unassigned)}"

    eval_visible = eval_df[EVAL_VISIBLE_COLUMNS].copy()
    eval_ground_truth = eval_df[["task_id"] + EVAL_HIDDEN_COLUMNS].copy()

    eval_visible.to_csv(OUTPUT_DIR / "evaluation_set.csv", index=False)
    eval_ground_truth.to_csv(OUTPUT_DIR / "evaluation_ground_truth.csv", index=False)

    # Guard: OOD evaluation descriptions must never appear in the repository.
    ood_descriptions = set(eval_df.loc[eval_df["template_source"] == "ood", "description"])
    leaked = ood_descriptions & set(repo_df["description"])
    assert not leaked, f"OOD contamination: {len(leaked)} descriptions also in the repository"

    # Validation report
    print(f"repository: {len(repo_df)} tasks, {repo_df['project_id'].nunique()} projects")
    print(f"evaluation: {len(eval_df)} tasks, {eval_df['project_id'].nunique()} projects")
    print(f"  in-distribution:     {(eval_df['template_source'] == 'repo').sum()}")
    print(f"  out-of-distribution: {(eval_df['template_source'] == 'ood').sum()}")
    print("  contamination check: passed")

    # Proportions rather than counts, and both sets side by side: what matters
    # is that the evaluation set mirrors the repository, not the raw totals.
    for column in ["category", "context_group", "priority", "variation_level"]:
        share = pd.DataFrame({
            "repository": repo_df[column].value_counts(normalize=True),
            "evaluation": eval_df[column].value_counts(normalize=True),
        }).fillna(0).round(3)
        print(f"\n-- {column} --")
        print(share.to_string())

    unique = repo_df["description"].nunique()
    with_deps = (repo_df["dependencies"] != "").sum()
    print(f"\nunique descriptions: {unique}/{len(repo_df)} ({unique / len(repo_df):.1%})")
    print(f"tasks with dependencies: {with_deps}/{len(repo_df)} ({with_deps / len(repo_df):.1%})")
    print(f"duration: {repo_df['estimated_duration'].min()}-"
          f"{repo_df['estimated_duration'].max()} min, "
          f"mean {repo_df['estimated_duration'].mean():.0f}")
    print(f"\nwritten to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()