"""
Synthetic dataset generator.

Builds projects made of chained tasks. Each task carries a natural-language
description (built from a template + domain vocabulary) plus the ground-truth
attributes the agent pipeline must later infer.

Two independent splits of the template pool exist:
  - "repo": templates used to build the historical repository (the knowledge base)
  - "ood" : templates held out from the repository, used only for evaluation tasks,
            so that "unseen description" genuinely means unseen. Backs RQ4.
"""

import random
from dataclasses import asdict          

from task import Task
from .vocab import DOMAIN_KEYS, pick_domain_vocab
from .templates import CATEGORIES

# ── Project structure ────────────────────────────────────────────────────
CORE_CHAIN = [
    "data_collection",
    "data_cleaning",
    "exploratory_data_analysis",
    "feature_engineering",
    "model_training",
    "model_evaluation",
    "reporting_documentation",
]
OPTIONAL_STAGES = {"feature_engineering", "exploratory_data_analysis"}
LOOP_STAGES = ["model_training", "model_evaluation"]

PRIORITY_LEVELS = {"low": 1, "medium": 2, "high": 3}

# Deadline slack (minutes of buffer after task duration) by priority: tighter for urgent work.
DEADLINE_SLACK_MINUTES = {"low": (240, 1440), "medium": (120, 480), "high": (30, 180)}

# ── Out-of-distribution template split ───────────────────────────────────
OOD_SEED = 20260717            
VARIATION_LEVELS = ["negation", "multi_step", "ambiguous"]


def _split_templates():
    """Hold out 2 templates per category for evaluation only: 1 standard +
    1 from a variation level, rotating the level across categories so that
    all four levels are represented in the OOD pool.

    Returns {category: {"repo": [(level, text), ...], "ood": [(level, text), ...]}}
    """
    rng = random.Random(OOD_SEED)
    splits = {}
    for i, (cat, meta) in enumerate(CATEGORIES.items()):
        levels = meta["templates"]

        var_level = VARIATION_LEVELS[i % len(VARIATION_LEVELS)]   # deterministic rotation
        held_out = {
            rng.choice(levels["standard"]),
            rng.choice(levels[var_level]),
        }

        repo, ood = [], []
        for lvl, texts in levels.items():
            for text in texts:
                (ood if text in held_out else repo).append((lvl, text))
        splits[cat] = {"repo": repo, "ood": ood}
    return splits


TEMPLATE_SPLITS = _split_templates()


def _sample_priority(weights: dict, rng: random.Random) -> str:
    labels = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(labels, weights=probs, k=1)[0]


def _fill_template(template: str, vocab: dict) -> str:
    return template.format(**vocab)


def _make_task(category, project_id, task_counter, domain_key, vocab, arrival_time,
               dependencies, rng, split="repo", id_prefix="task"):
    meta = CATEGORIES[category]
    pool = TEMPLATE_SPLITS[category][split]
    level, template = rng.choice(pool)            # pool holds (level, text) tuples
    description = _fill_template(template, vocab)

    lo, hi = meta["duration_range"]
    duration = rng.randint(lo, hi)

    priority = _sample_priority(meta["priority_weights"], rng)
    slack_lo, slack_hi = DEADLINE_SLACK_MINUTES[priority]
    deadline = arrival_time + duration + rng.randint(slack_lo, slack_hi)

    return Task(
        task_id=f"{id_prefix}_{task_counter:06d}",
        project_id=project_id,
        description=description,
        domain=domain_key,
        category=category,
        context_group=meta["context_group"],
        estimated_duration=duration,
        priority=priority,
        priority_level=PRIORITY_LEVELS[priority],
        arrival_time=arrival_time,
        deadline=deadline,
        variation_level=level,
        dependencies=list(dependencies),
        template_source=split,
    )


def _build_project_chain(rng: random.Random) -> list:
    """Decide which core stages this project includes, keeping order fixed."""
    stages = []
    for stage in CORE_CHAIN:
        if stage in OPTIONAL_STAGES and rng.random() < 0.25:
            continue
        stages.append(stage)

    # Sometimes repeat the modelling loop (iterative experimentation)
    if rng.random() < 0.4 and "model_training" in stages:
        n_extra_loops = rng.randint(1, 2)
        insert_at = (stages.index("reporting_documentation") if "reporting_documentation" in stages else len(stages))
        loop_block = [s for s in LOOP_STAGES if s in stages]
        stages = stages[:insert_at] + loop_block * n_extra_loops + stages[insert_at:]
    return stages


def generate_projects(n_target_tasks: int, rng: random.Random, split="repo",
                      horizon_minutes=60 * 24 * 60, ood_fraction=0.3,
                      id_prefix="task", project_prefix="proj"):
    """
    Generate projects (and their tasks) until roughly n_target_tasks tasks exist.

    split="repo"  -> every task uses repository templates.
    split="eval"  -> each task independently draws an OOD template with
                     probability `ood_fraction`, otherwise a repo template.

    `horizon_minutes` is the scheduling horizon (default: 60 days).
    `id_prefix`/`project_prefix` keep repository and evaluation ids disjoint.
    """
    tasks = []
    project_idx = 0

    def pick_split():
        if split == "repo":
            return "repo"
        return "ood" if rng.random() < ood_fraction else "repo"

    while len(tasks) < n_target_tasks:
        project_idx += 1
        project_id = f"{project_prefix}_{project_idx:05d}"
        domain_key = rng.choice(DOMAIN_KEYS)
        vocab = pick_domain_vocab(domain_key, rng)

        stages = _build_project_chain(rng)
        project_start = rng.randint(0, max(horizon_minutes - 7 * 24 * 60, 1))
        cursor = project_start
        prev_task_id = None

        for stage in stages:
            cursor += rng.randint(15, 24 * 60)   

            deps = [prev_task_id] if (stage != CORE_CHAIN[0] and prev_task_id) else []

            t = _make_task(
                category=stage, project_id=project_id, task_counter=len(tasks) + 1,
                domain_key=domain_key, vocab=vocab, arrival_time=cursor,
                dependencies=deps, rng=rng, split=pick_split(), id_prefix=id_prefix,
            )
            tasks.append(t)
            prev_task_id = t.task_id
            if len(tasks) >= n_target_tasks:
                break

        if len(tasks) < n_target_tasks:
            for _ in range(rng.randint(0, 3)):
                t = _make_task(
                    category="data_pipeline_maintenance", project_id=project_id,
                    task_counter=len(tasks) + 1, domain_key=domain_key,
                    vocab=vocab,                                   
                    arrival_time=rng.randint(project_start, project_start + 14 * 24 * 60),
                    dependencies=[], rng=rng, split=pick_split(), id_prefix=id_prefix,
                )
                tasks.append(t)
                if len(tasks) >= n_target_tasks:
                    break

    return tasks[:n_target_tasks]


def tasks_to_records(tasks):
    records = []
    for t in tasks:
        d = asdict(t)
        d["dependencies"] = "|".join(d["dependencies"]) if d["dependencies"] else ""
        records.append(d)
    return records