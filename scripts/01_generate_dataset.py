"""
Generate the synthetic dataset: historical repository + evaluation set (+ hidden
ground truth for the evaluation set).
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset.generator import generate_projects, tasks_to_records 

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repository", type=int, default=4500)
    parser.add_argument("--n-evaluation", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Historical repository (full attributes, repo-only templates)
    repo_tasks = generate_projects(
        args.n_repository, rng, split="repo", id_prefix="task", project_prefix="proj"
    )
    repo_df = pd.DataFrame(tasks_to_records(repo_tasks))
    repo_df.to_csv(OUTPUT_DIR / "historical_repository.csv", index=False)

    # Evaluation set (mix of in-distribution + out-of-distribution templates)
    eval_rng = random.Random(args.seed + 1)  # different stream, still reproducible
    eval_tasks = generate_projects(
        args.n_evaluation, eval_rng, split="eval", id_prefix="evaltask", project_prefix="evalproj"
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
    print("=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Historical repository: {len(repo_df)} tasks, {repo_df['project_id'].nunique()} projects")
    print(f"Evaluation set:        {len(eval_df)} tasks, {eval_df['project_id'].nunique()} projects")
    print(f"  - in-distribution (repo templates): {(eval_df['template_source'] == 'repo').sum()}")
    print(f"  - out-of-distribution (unseen templates): {(eval_df['template_source'] == 'ood').sum()}")
    print(f"  - OOD contamination check: passed (0 overlapping descriptions)")
    print()
    print("-- Repository: category distribution --")
    print(repo_df["category"].value_counts())
    print()
    print("-- Repository: context_group distribution --")
    print(repo_df["context_group"].value_counts())
    print()
    print("-- Repository: domain distribution --")
    print(repo_df["domain"].value_counts())
    print()
    print("-- Repository: priority distribution --")
    print(repo_df["priority"].value_counts())
    print()
    print("-- Variation levels (proposal §9) --")
    levels = pd.DataFrame({
        "repository": repo_df["variation_level"].value_counts(),
        "evaluation": eval_df["variation_level"].value_counts(),
    }).fillna(0).astype(int)
    levels["repo %"] = (levels["repository"] / len(repo_df) * 100).round(1)
    levels["eval %"] = (levels["evaluation"] / len(eval_df) * 100).round(1)
    print(levels)
    print()
    print(f"Unique descriptions in repository: {repo_df['description'].nunique()} / {len(repo_df)} "
          f"({repo_df['description'].nunique() / len(repo_df):.1%} unique)")
    print(f"Duration range: {repo_df['estimated_duration'].min()}-{repo_df['estimated_duration'].max()} min, "
          f"mean {repo_df['estimated_duration'].mean():.1f} min")
    n_with_deps = (repo_df["dependencies"] != "").sum()
    print(f"Tasks with dependencies: {n_with_deps} / {len(repo_df)} ({n_with_deps / len(repo_df):.1%})")
    print()
    print("Sample tasks:")
    with pd.option_context("display.max_colwidth", 90):
        print(repo_df[["description", "category", "context_group",
                       "variation_level", "estimated_duration", "priority",
                       "dependencies"]].sample(8, random_state=1).to_string())
    print()
    print(f"Files written to: {OUTPUT_DIR}")
    print(f"  historical_repository.csv    ({len(repo_df.columns)} columns)")
    print(f"  evaluation_set.csv           ({len(eval_visible.columns)} columns, visible only)")
    print(f"  evaluation_ground_truth.csv  ({len(eval_ground_truth.columns)} columns, hidden)")


if __name__ == "__main__":
    main()