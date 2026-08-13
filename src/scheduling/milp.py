"""
MILP scheduler — the proposed system (PuLP + CBC).

Objective, three weighted terms: (needed to be adjust)
    W_MAKESPAN            * makespan
    W_CONTEXT_SWITCH      * context switching cost
    W_DEADLINE_VIOLATION  * total lateness

Formulation: positional. x[i][p] = 1 iff task i runs in position p of the
sequence. A single executor runs tasks in series, so there are exactly n
positions for n tasks. Think of x as an n x n table in which the solver must
place one 1 per row and one 1 per column; the location of those 1s is the order.

Like every baseline, this module only decides the ORDER. Times are then
assigned by sequence_to_schedule(), identically for all methods, so the
comparison stays fair.
"""

import time
import pulp
import config
from scheduling.schedule import sequence_to_schedule


def _big_m(tasks, switch_cost: int) -> int:
  """An upper bound on any time in this instance.

  Big-M must be large enough never to bind, but as TIGHT as possible: an
  oversized M weakens the LP relaxation and causes numerical instability,
  which makes CBC crawl. Latest arrival + all durations + all possible
  switches cannot be exceeded by any feasible schedule.
  """
  return (max(t.arrival_time for t in tasks)
          + sum(t.estimated_duration for t in tasks)
          + len(tasks) * switch_cost)


def build_model(tasks: list,
                switch_cost: int = config.CONTEXT_SWITCH_COST_MIN,
                w_makespan: float = config.W_MAKESPAN,
                w_switch: float = config.W_CONTEXT_SWITCH,
                w_lateness: float = config.W_DEADLINE_VIOLATION):
  """Build the MILP. Returns (model, x) so the caller can read the solution."""
  n = len(tasks)
  positions = range(n)
  indices = range(n)
  big_m = _big_m(tasks, switch_cost)

  model = pulp.LpProblem("task_scheduling", pulp.LpMinimize)

  # Variables
  x = pulp.LpVariable.dicts("x", (indices, positions), cat="Binary")

  # s[p] = 1 if position p has a different context_group than position p-1.
  # Position 0 has no predecessor, so it can never be a switch.
  s = pulp.LpVariable.dicts("s", range(1, n), cat="Binary")

  start = pulp.LpVariable.dicts("start", positions, lowBound=0)
  end = pulp.LpVariable.dicts("end", positions, lowBound=0)
  makespan = pulp.LpVariable("makespan", lowBound=0)

  # lowBound=0 is what implements max(0, overrun): finishing early cannot
  # produce negative lateness that offsets another task's delay.
  lateness = pulp.LpVariable.dicts("lateness", positions, lowBound=0)

  # Objective
  model += (w_makespan * makespan
            + w_switch * switch_cost * pulp.lpSum(s[p] for p in range(1, n))
            + w_lateness * pulp.lpSum(lateness[p] for p in positions)), "cost"

  # Assignment: one position per task, one task per position
  for i in indices:
      model += pulp.lpSum(x[i][p] for p in positions) == 1, f"task_{i}_once"
  for p in positions:
      model += pulp.lpSum(x[i][p] for i in indices) == 1, f"position_{p}_once"

  # Context switches
  # "if task i is at p AND task j is at p-1 AND contexts differ, then s[p]=1"
  # becomes  s[p] >= x[i][p] + x[j][p-1] - 1.  When both are 1 the right side
  # forces s[p]=1; otherwise it is <= 0 and leaves s[p] free — and since the
  # objective is minimised, the solver sets it to 0 by itself.
  for p in range(1, n):
      for i in indices:
          for j in indices:
              if i != j and tasks[i].context_group != tasks[j].context_group:
                  model += s[p] >= x[i][p] + x[j][p - 1] - 1, f"switch_{p}_{i}_{j}"

  # Times
  # Only one x[i][p] is 1, so this sum collapses to the duration of whichever
  # task lands in p. Constant * variable, hence linear.
  for p in positions:
      duration_p = pulp.lpSum(tasks[i].estimated_duration * x[i][p] for i in indices)
      model += end[p] == start[p] + duration_p, f"end_{p}"

  # Positions run in series, paying the switch cost in between.
  for p in range(1, n):
      model += start[p] >= end[p - 1] + switch_cost * s[p], f"chain_{p}"

  # Arrival, via Big-M: "if task i is at p then start[p] >= arrival_i".
  # x=1 -> the M term vanishes and the constraint bites.
  # x=0 -> the right side goes hugely negative and the constraint is inert.
  for p in positions:
      for i in indices:
          model += (start[p] >= tasks[i].arrival_time - big_m * (1 - x[i][p]),
                    f"arrival_{i}_{p}")

  # Makespan as >= every end; minimising pushes it down to the true maximum.
  for p in positions:
      model += makespan >= end[p], f"makespan_ge_{p}"

  # Lateness against the deadline of whichever task lands in p
  for p in positions:
      deadline_p = pulp.lpSum(tasks[i].deadline * x[i][p] for i in indices)
      model += lateness[p] >= end[p] - deadline_p, f"lateness_{p}"

  # Dependencies (hard)
  # position_of(i) is linear: each position INDEX times its binary, and only
  # one binary is 1. No Big-M needed — this constraint always applies.
  task_index = {t.task_id: i for i, t in enumerate(tasks)}

  def position_of(i):
      return pulp.lpSum(p * x[i][p] for p in positions)

  for i, task in enumerate(tasks):
      for dep_id in task.dependencies:
          if dep_id not in task_index:
              continue      # dependency outside this batch: assumed already done
          d = task_index[dep_id]
          model += (position_of(d) + 1 <= position_of(i),
                    f"dep_{dep_id}_before_{task.task_id}")

  return model, x


def solve(tasks: list, time_limit: int = config.SOLVER_TIME_LIMIT_S, **weights):
  """Optimise the execution order and return (schedule, info).

  info carries the solver status and wall-clock solve time, both of which
  belong in the report: on large batches CBC may hit the time limit and
  return the best feasible solution found rather than a proven optimum.
  """
  if not tasks:
      return sequence_to_schedule([]), {"status": "Empty", "solve_time_s": 0.0}

  model, x = build_model(tasks, **weights)

  t0 = time.perf_counter()
  model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
  solve_time = time.perf_counter() - t0

  status = pulp.LpStatus[model.status]
  info = {
      "status": status,
      "solve_time_s": round(solve_time, 2),
      "objective": pulp.value(model.objective),
      "n_tasks": len(tasks),
  }

  if status != "Optimal":
      # Infeasible / Undefined / Not Solved: no usable assignment to read.
      return sequence_to_schedule([]), info

  # Read the order back out of the binaries.
  order = [None] * len(tasks)
  for i in range(len(tasks)):
      for p in range(len(tasks)):
          value = x[i][p].varValue
          if value is not None and round(value) == 1:
              order[p] = tasks[i]

  return sequence_to_schedule(order), info


def re_optimize(pending_tasks: list, new_task, clock: int = 0, **weights):
  """Dynamic re-optimisation when a new task arrives mid-execution.

  Completed tasks are fixed and simply dropped; whatever is still pending,
  plus the newcomer, is re-solved from the current clock. Callers keep the
  previous schedule so the UI can show what changed and why.
  """
  tasks = list(pending_tasks) + [new_task]
  for t in tasks:                       # nothing may start before "now"
      t.arrival_time = max(t.arrival_time, clock)
  return solve(tasks, **weights)