# A Multi-Agent Decision Support System for Task Scheduling in Data Science Workflows

MSc Data Science and Artificial Intelligence — Final Project
Birkbeck, University of London

---

## What this project does

Data science practitioners receive work as incomplete natural language: *"clean the churn dataset before training the model"*, *"the numbers look off, look into it"*. Nobody states how long a task will take, how urgent it is, or what kind of work it involves — yet these tasks still have to be ordered, respecting dependencies, deadlines, and the real cost of constantly switching between different kinds of work.

This system addresses both halves of that problem with two different technologies:

**Interpretation** — an LLM agent pipeline reads each description and estimates the missing attributes (duration, priority, category, type of work).

**Decision** — a Mixed-Integer Linear Programming model takes those attributes and computes an optimised execution order.

The separation is deliberate: **the language model never decides the order, and the optimiser never interprets language**. Each component does what it is good at.

```
Natural language descriptions
            │
            ▼
  [1] Retrieval Agent ────── finds the k most similar historical tasks
            │
            ▼
  [2] Inference Agent ────── estimates category, duration, priority, context
            │
            ▼
  [3] Grouping Agent ─────── detects groupings across the batch
            │
            ▼
     Complete tasks
            │
   ┌────────┴────────┐
   ▼                 ▼
 MILP           5 baselines
(PuLP + CBC)  (FIFO, STF, EDF, Priority, Agent-only)
   │                 │
   └────────┬────────┘
            ▼
   sequence_to_schedule()   ← turns ORDER into TIMES, identically for all methods
            │
            ▼
    Schedule + metrics
```

---

## Current status

| Component | Status |
|---|---|
| Synthetic dataset generation | **Working** |
| Data loading (CSV to objects) | **Working** |
| Schedule format and timing model | **Working** |
| Scheduling metrics | **Working** |
| Baselines: FIFO, STF, EDF, Priority | **Working** |
| MILP optimiser | In progress |
| LLM access layer with caching | In progress |
| Retrieval / Inference / Grouping agents | In progress |
| Agent-only baseline | In progress |
| Inference metrics (MAE, F1) | In progress |
| Full evaluation and Streamlit demo | In progress |

The script that run end to end today: `01_generate_dataset.py`

---

## Setup

Requires Python 3.10 or later.

```bash
git clone https://github.com/sotomarcos24/MSc-Project.git
cd MSc-Project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The MILP solver (CBC) ships with PuLP, so no separate installation is needed.

**Important — always run scripts from the repository root:**

```bash
python scripts/01_generate_dataset.py          
```

Each script computes its paths relative to its own location and expects the working directory to be the repository root.

An OpenAI API key is only needed for the agent components, which are still in progress. When required, copy `.env.example` to `.env` and add `OPENAI_API_KEY=sk-...`. The datasets and the scheduling pipeline run with no API key at all.

---

## Script 1 — Generating the dataset

```bash
python scripts/01_generate_dataset.py
```

Optional arguments:

```bash
python scripts/01_generate_dataset.py --n-repository 4500 --n-evaluation 600 --seed 42
```

### What it does

There is no public dataset of natural-language data science tasks annotated with duration, priority and workflow stage, so the project generates its own. The generator is written entirely in Python from hand-designed sentence templates — **deliberately not using an LLM**, which would introduce circular bias into a system whose purpose is to evaluate LLM inference.

The generator builds **whole projects**, not isolated tasks. Each project picks an application domain (customer churn, fraud detection, sales forecasting…), fixes a vocabulary so all its tasks talk about the same dataset, and then walks a canonical data science workflow: collection → cleaning → exploratory analysis → feature engineering → training → evaluation → reporting. Each stage depends on the previous one, some stages are skipped, and the modelling block is sometimes repeated to simulate iterative experimentation. Independent maintenance tasks are scattered across the timeline as interruptions.

### The label space

Each task belongs to one of **8 categories**, which map onto **5 context groups** — the cognitive stage of work:

| Context group | Categories |
|---|---|
| `data_preparation` | data_collection, data_cleaning |
| `analysis` | exploratory_data_analysis, feature_engineering |
| `modelling` | model_training, model_evaluation |
| `reporting` | reporting_documentation |
| `infrastructure` | data_pipeline_maintenance |

Category and context group are **different attributes**. Several categories share a stage, so moving from collection to cleaning costs no context switch, while moving from cleaning to reporting does. This is what the optimiser exploits.

### Linguistic variation

Descriptions are drawn from 152 templates spanning **four levels of variation**, 19 per category:

| Level | Per category | Example |
|---|---|---|
| `standard` | 10 | *"Clean missing values in the {dataset} before training the {model}."* |
| `negation` | 3 | *"Resolve inconsistent entries in the {dataset} without dropping any rows."* |
| `multi_step` | 3 | *"Deduplicate the {dataset} and then standardise the {feature} column formats."* |
| `ambiguous` | 3 | *"The {dataset} looks messy again, deal with it."* |

Placeholders are filled from 15 application domains, each supplying its own datasets, targets, models and features.

### Out-of-distribution split

Two templates per category are **reserved exclusively for evaluation tasks** and never used when building the historical repository, so that "unseen description" means genuinely unseen. The held-out pair is always one standard template plus one variation template, with the variation level rotating deterministically across categories so all four levels are represented.

This split uses its own fixed seed, independent of the command-line `--seed`, because it must stay identical across runs: if it changed, regenerating only the evaluation set could produce "unseen" templates that are in fact present in the repository — silent contamination that would invalidate the generalisation results.

### Output

Three files in `data/raw/`.

**`historical_repository.csv`** — 4,500 tasks with every attribute known. This is the knowledge base the retrieval agent searches.

**`evaluation_set.csv`** — 600 tasks, containing **only what the system is allowed to see**:

| Column | Meaning |
|---|---|
| `task_id` | Unique identifier |
| `description` | The natural language text |
| `arrival_time` | Minutes from the start of the horizon |
| `deadline` | Minutes from the start of the horizon |

**`evaluation_ground_truth.csv`** — the same 600 tasks with the hidden attributes, used only for scoring:

| Column | Meaning |
|---|---|
| `project_id` | Which project the task belongs to (hidden: the grouping agent must infer it) |
| `domain` | Application domain — known context, not something the agents infer |
| `category` | One of the 8 task categories |
| `context_group` | One of the 5 cognitive stages |
| `estimated_duration` | Minutes |
| `priority` / `priority_level` | `low`/`medium`/`high` and its numeric mirror 1/2/3 |
| `variation_level` | standard / negation / multi_step / ambiguous |
| `dependencies` | Pipe-separated task ids, e.g. `evaltask_000001` |
| `template_source` | `repo` or `ood` |

`variation_level` and `template_source` are analysis metadata: they let the evaluation break inference accuracy down by linguistic difficulty and by whether the description had been seen before.

All time values are **integer minutes from the start of the horizon**, not dates. This keeps the optimisation model and the metrics simple; conversion to human-readable times happens only in the visualisation layer.

### Verifying the output

The script prints a validation report: task counts, distributions by category, context group, domain, priority and variation level, the proportion of unique descriptions, and a sample of generated tasks.

It also runs two assertions that protect the validity of the experiment. The first checks that every generated column has been explicitly classified as visible or hidden, so adding a field later cannot silently drop it from both files. The second checks that no description marked as out-of-distribution also appears in the historical repository.

Typical output with the default seed:

```
Historical repository: 4500 tasks, 497 projects
Evaluation set:        600 tasks, 66 projects
  - in-distribution (repo templates): 436
  - out-of-distribution (unseen templates): 164
  - OOD contamination check: passed (0 overlapping descriptions)

-- Variation levels --
                 repository  evaluation  repo %  eval %
standard               2393         300    53.2    50.0
multi_step              721         115    16.0    19.2
negation                700          95    15.6    15.8
ambiguous               686          90    15.2    15.0

Unique descriptions in repository: 3250 / 4500 (72.2% unique)
```

Generation is fully deterministic: the same seed produces identical files on any machine.

The generated CSVs are committed to the repository, so this script only needs to be run again if the templates or the generator change.

---