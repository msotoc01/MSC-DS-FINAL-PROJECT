"""Step 4 — The evaluation that produces the report's results.

Runs every scheduler over several backlogs, in both attribute modes, and prints
the comparison. The experiment parameters are fixed constants below rather than
command-line options: the evaluation should be the same experiment every time,
so that no table can be produced with different settings by accident.
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
from agents.pipeline import enrich_tasks
from dataset.loaders import load_evaluation_tasks
from evaluation import metrics
from scheduling import milp
from scheduling.baselines import ALL_BASELINES
from scheduling.schedule import check_dependencies, sequence_to_schedule
from scheduling.backlog import build_batch

# experiment parameters 
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56]
WORK_DAYS = 2          # keeps backlogs in the 10-15 task range where CBC converges

METRICS = ["makespan", "context_switches", "deadline_satisfaction",
           "total_lateness", "dependency_violations"]
METHOD_ORDER = list(ALL_BASELINES) + ["MILP"]


def with_inferred(task, source):
    """A copy of task carrying the agents' estimates instead of the real ones.

    arrival_time, deadline and dependencies are untouched: they come from the
    dataset and are the planning problem itself, not something to estimate.
    """
    clone = copy.deepcopy(task)
    clone.category = source.category
    clone.context_group = source.context_group
    clone.estimated_duration = source.estimated_duration
    clone.priority = source.priority
    clone.priority_level = source.priority_level
    return clone


def summarise(name, order, real_batch, seed, mode):
    """Time an ordering with the REAL durations and measure it.

    The schedulers decide the order from whatever attributes they were given;
    this re-times that order against reality, so both modes are measured on the
    same footing.
    """
    by_id = {t.task_id: t for t in real_batch}
    real_order = [by_id[t.task_id] for t in order]
    schedule = sequence_to_schedule(real_order)
    return {
        "seed": seed, "mode": mode, "method": name, "n_tasks": len(real_batch),
        "makespan": metrics.makespan(schedule),
        "context_switches": metrics.context_switches(schedule),
        "deadline_satisfaction": metrics.deadline_satisfaction(schedule),
        "total_lateness": metrics.total_lateness(schedule),
        "dependency_violations": len(check_dependencies(real_order)),
    }


def run_batch(batch, real_batch, seed, mode):
    """Every scheduler on one backlog. Returns one row per method."""
    by_id = {t.task_id: t for t in batch}

    # Each scheduler gets its own copy: sorting is harmless, but a shared list
    # would make the run order matter, and that must never happen.
    rows = [summarise(name, [by_id[i] for i in run(list(batch)).task_id],
                      real_batch, seed, mode)
            for name, run in ALL_BASELINES.items()]

    schedule, info = milp.solve(list(batch))
    row = summarise("MILP", [by_id[i] for i in schedule.task_id],
                    real_batch, seed, mode)
    # CBC reports 'Optimal' even after exhausting its time limit, so a run that
    # stopped at the limit is flagged rather than trusted.
    row["solve_time_s"] = info["solve_time_s"]
    row["proven_optimal"] = (info["status"] == "Optimal" and
                             info["solve_time_s"] < config.SOLVER_TIME_LIMIT_S * 0.98)
    rows.append(row)
    return rows


def main():
    seeds = [int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    runs = config.ROOT_DIR / "results"
    runs.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.name.split("_")[1]) for p in runs.glob("run_*")), default=0)
    out_dir = runs / f"run_{n:03d}"
    out_dir.mkdir()

    print(f"seeds {seeds} | horizon {WORK_DAYS}d\n")

    tasks = load_evaluation_tasks(with_ground_truth=True)
    batches = {s: build_batch(copy.deepcopy(tasks), work_days=WORK_DAYS, seed=s)
               for s in seeds}

    # Only the tasks that actually end up in a backlog are sent to the agents:
    # a handful of batches means tens of calls instead of six hundred.
    used = {t.task_id: t for batch in batches.values() for t in batch}
    print(f"inferring attributes for {len(used)} tasks used across the backlogs")
    inferred, log = enrich_tasks(copy.deepcopy(list(used.values())),
                                 pd.read_csv(config.HISTORICAL_CSV))
    inferred = {t.task_id: t for t in inferred}

    rows = []
    for mode in ["oracle", "inferred"]:
        print(f"\n[{mode}]")
        for seed, batch in batches.items():
            planning = (batch if mode == "oracle"
                        else [with_inferred(t, inferred[t.task_id]) for t in batch])
            print(f"  seed {seed}: {len(batch)} tasks", end=" ", flush=True)
            rows.extend(run_batch(planning, batch, seed, mode))
            print("done")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_results.csv", index=False)
    log.to_csv(out_dir / "inference_log.csv", index=False)

    for mode in ["oracle", "inferred"]:
        print(f"\n-- {mode} --  mean over {len(seeds)} seeds")
        block = raw[raw["mode"] == mode].groupby("method")[METRICS].mean().round(2)
        print(block.loc[[m for m in METHOD_ORDER if m in block.index]].to_string())

    # Both modes are timed with the real durations, so this is the cost of
    # planning from estimates instead of from the truth.
    means = raw.groupby(["mode", "method"])[METRICS].mean()
    gap = (means.loc["inferred"] - means.loc["oracle"]).round(2)
    print("\n-- cost of inference (inferred - oracle) --")
    print(gap.loc[[m for m in METHOD_ORDER if m in gap.index]].to_string())

    # The column is object dtype (baseline rows leave it empty), so it has to be
    # cast before negating: ~ on an object column counts bitwise, not logically.
    proven = raw.loc[raw["method"] == "MILP", "proven_optimal"].fillna(False).astype(bool)
    if not proven.all():
        print(f"\nWARNING: {(~proven).sum()} of {len(proven)} MILP runs hit the "
              "time limit — feasible but not proven optimal.")

    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()