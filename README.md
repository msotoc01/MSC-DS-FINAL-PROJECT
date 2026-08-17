# A Multi-Agent Decision Support System for Task Scheduling in Data Science Workflows

MSc Data Science and Artificial Intelligence — Final Project
Birkbeck, University of London

Author: Marcos Soto

> **Work in progress.** This repository is an intermediate snapshot, not a finished system. Nothing here is final: the datasets, the parameters and several design choices are still under review and are likely to change. Roughly half the system is still to be built.

---

## The idea

Data science work arrives as incomplete natural language: *"clean the churn dataset before training the model"*, *"the numbers look off, look into it"*. Nobody says how long a task will take, how urgent it is, or what kind of work it involves — but the tasks still have to be ordered, respecting dependencies, deadlines, and the cost of switching between different kinds of work.

The system splits that into two layers:

- **Interpretation** — LLM agents read each description and estimate the missing attributes.
- **Decision** — a MILP model (PuLP + CBC) takes those attributes and computes the execution order.

The language model never decides the order, and the optimiser never interprets language.

```
descriptions → retrieval → inference → grouping → complete tasks
                                                       │
                                          MILP  ←──────┴──────→  baselines
                                             │                      │
                                             └────→ schedule ←──────┘
                                                        │
                                                    metrics
```

---

## What works and what does not

| | |
|---|---|
| Dataset generation and loading | Working (changes may be made)|
| Timing model and schedule format | Working (changes may be made)|
| Baselines (FIFO, STF, EDF, Priority) | Working |
| MILP optimiser | In Progress |
| Scheduling metrics | Working |
| LLM access layer | Started |
| Retrieval / Inference / Grouping agents | Not yet built |
| Agent-only baseline, inference metrics | Not yet built |
| Streamlit demo | Not yet built |

Two scripts run end to end today. Neither needs an API key.

---

## Running it

Python 3.10 or later. Always run from the repository root.

```bash
git clone https://github.com/msotoc01/MSC-DS-FINAL-PROJECT.git
cd MSC-DS-FINAL-PROJECT

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/01_generate_dataset.py        # regenerates the CSVs (already committed)
python scripts/02_run_pipeline_sample.py     # compares MILP against the baselines
```

---

## The two scripts

**`01_generate_dataset.py`** builds the synthetic dataset, since no public dataset of natural-language data science tasks annotated with duration, priority and workflow stage exists. Descriptions come from hand-written templates rather than an LLM, to avoid circular bias in a system whose purpose is to evaluate LLM inference. It produces three files in `data/raw/`: a historical repository of 4,500 tasks with all attributes known, an evaluation set of 600 tasks showing only what the system is allowed to see, and the matching hidden ground truth. Generation is deterministic for a given seed.

**`02_run_pipeline_sample.py`** builds a batch and runs every scheduler on it, using ground-truth attributes so that any difference between methods comes from the scheduling decision alone. A batch is treated as a backlog — work already waiting at the start of a planning period — rather than a stream of arrivals. All schedulers decide only the order; times are then assigned by one shared function, so the comparison stays fair.

---

## Repository layout

```
config.py                    parameters
data/raw/                    the three generated CSVs
scripts/                     the two runnable scripts
src/task.py                  the shared data model
src/dataset/                 generation and loading
src/scheduling/              schedule format, baselines, MILP
src/evaluation/metrics.py    metrics
src/agents/                  LLM layer and agents
dashboard/                   Streamlit demo
```

---

## Open questions

These are known and being worked on:

- The deadline formula currently derives the deadline from the arrival time, which makes the two almost perfectly correlated and gives FIFO an unearned advantage. Being reworked.
- CBC can report a solution as optimal after exhausting its time limit, so solver status needs checking against elapsed time before results can be trusted on larger batches.
- Batch size, overload factor and the number of projects per batch are provisional values, not calibrated ones.