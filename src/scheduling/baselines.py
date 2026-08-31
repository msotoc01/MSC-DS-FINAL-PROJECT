"""
The 5 baseline schedulers.

All of them decide only the ORDER; times are assigned by
sequence_to_schedule(), identically for every method, so the comparison is fair.
"""

import config
from scheduling.schedule import sequence_to_schedule
from task import Task
from agents.llm import ask, parse_json


def fifo(tasks: list[Task]):
    """First in, first out: earliest arrival first."""
    ordered = sorted(tasks, key=lambda t: t.arrival_time)
    return sequence_to_schedule(ordered)


def shortest_task_first(tasks: list[Task]):
    """Shortest estimated duration first."""
    ordered = sorted(tasks, key=lambda t: t.estimated_duration)
    return sequence_to_schedule(ordered)


def earliest_deadline_first(tasks: list[Task]):
    """Earliest deadline first."""
    ordered = sorted(tasks, key=lambda t: t.deadline)
    return sequence_to_schedule(ordered)


def priority_first(tasks: list[Task]):
    """Highest priority first; ties broken by earliest deadline."""
    ordered = sorted(tasks, key=lambda t: (-t.priority_level, t.deadline))
    return sequence_to_schedule(ordered)


AGENT_SYSTEM = """\
You are an experienced data science team lead deciding the order in which one
person will work through a queue of tasks today.
 
You are not asked to describe a plan or to explain your reasoning. You are
asked for the sequence itself, and it has to be executable: every task appears
exactly once, and nothing runs before the work it depends on.
 
Think about the trade-off before you commit. Grouping tasks of the same kind
saves the time lost to switching, but a task with an early deadline may not be
able to wait for its group. Neither rule wins on its own.
 
Reply with one JSON array and nothing else: no explanation, no code fences."""
 
AGENT_PROMPT = """\
Order these {n} tasks for one person working through them one at a time.
 
id | duration | context group | deadline | dependencies
{lines}
 
All tasks are available now. Times are in working minutes from now, so a
deadline of 480 means the end of the first working day.
 
Total work: {total} minutes. Every switch between context groups costs a
further {switch_cost} minutes, which delays everything after it.
 
Optimise, in this order of importance:
1. Never place a task before something it depends on. This is not a
   preference: a schedule that breaks it cannot be executed.
2. Miss as few deadlines as possible, and where a deadline must be missed,
   miss it by as little as possible.
3. Group tasks of the same context group together, to pay fewer switches.
4. Finish as early as possible.
 
There is more work here than fits comfortably, so some deadlines may be
impossible to meet. Order the tasks anyway.
 
Reply with only a JSON array of all {n} task ids, in execution order:
["first_id", "second_id", ...]"""


def agent_only(tasks: list[Task]):
    """The language model orders the batch directly, with no optimiser.
 
    This is the alternative architecture to the whole system: give an LLM the
    tasks and ask for a schedule. 
 
    The model sees exactly what the optimiser sees — duration, context group,
    deadline and dependencies — so any difference is in the decision, not in
    the information available.
    """
 
    lines = "\n".join(
        f"- {t.task_id} | {t.estimated_duration} min | {t.context_group} | "
        f"due {t.deadline}"
        + (f" | after: {', '.join(t.dependencies)}" if t.dependencies else "")
        for t in tasks
    )
    total = sum(t.estimated_duration for t in tasks)
    prompt = AGENT_PROMPT.format(n=len(tasks), lines=lines, total=total,
                                 switch_cost=config.CONTEXT_SWITCH_COST_MIN)
 
    reply = parse_json(ask(prompt, system=AGENT_SYSTEM))
 
    by_id = {t.task_id: t for t in tasks}
    ordered, seen = [], set()
    for task_id in (reply or []):
        if task_id in by_id and task_id not in seen:
            ordered.append(by_id[task_id])
            seen.add(task_id)
    # Anything the model dropped still has to be executed.
    ordered += [t for t in sorted(tasks, key=lambda t: t.arrival_time)
                if t.task_id not in seen]
 
    return sequence_to_schedule(ordered)
 
 
ALL_BASELINES = {
    "FIFO": fifo,
    "STF": shortest_task_first,
    "EDF": earliest_deadline_first,
    "Priority": priority_first,
    "Agent-only": agent_only,
}
