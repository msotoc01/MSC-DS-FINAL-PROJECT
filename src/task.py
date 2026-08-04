"""
Core data model: the Task.

This is the contract of the whole system. The dataset generator produces Tasks,
the agent pipeline enriches them, the schedulers consume them and the evaluation
scores them. Every module imports Task from here — never redefine it locally.

The CSV column names produced by tasks_to_records() are exactly these field
names; keep them in sync or nothing downstream will read the data correctly.
"""

from dataclasses import dataclass, field

@dataclass
class Task:
    task_id: str
    project_id: str
    description: str
    domain: str                                        # project domain (churn, fraud...) — known, not inferred
    category: str                                      # one of the 8 task categories — inferred
    context_group: str                                 # cognitive workflow stage — inferred, drives switching cost
    estimated_duration: int                            # minutes — inferred
    priority: str                                      # low/medium/high — inferred
    priority_level: int                                # numeric mirror of priority, for the MILP
    arrival_time: int                                  # minutes from horizon start
    deadline: int                                      # minutes from horizon start
    variation_level: str                               # standard/negation/multi_step/ambiguous — for error analysis
    dependencies: list = field(default_factory=list)
    template_source: str = "repo"                      # "repo" or "ood" — only meaningful for evaluation tasks