"""
Agent 2 — Attribute Inference.

Input:  a task description + the k similar past tasks found by the retrieval agent.
Output: category, context_group, estimated duration, priority.
"""

from dataset.templates import CATEGORIES, CATEGORY_KEYS
from agents.llm import ask, parse_json

# category -> context_group, taken from the single definition in templates.py
# rather than duplicated here, so the two can never drift apart.
CATEGORY_TO_CONTEXT = {name: meta["context_group"] for name, meta in CATEGORIES.items()}

PRIORITY_LEVELS = {"low": 1, "medium": 2, "high": 3}

SYSTEM = """\
You are an experienced data science team lead estimating the attributes of
incoming work from short, often incomplete task descriptions.

You are shown similar past tasks with their recorded attributes. Treat them as
evidence, not as answers to copy: tasks described in almost identical words
routinely differ in how long they take.

When the description gives you no reason to think a task is unusually large or
small, estimate the typical value for its kind. Confident outliers are worse
than calibrated middles.

Reply with one JSON object and nothing else: no explanation, no code fences."""

PROMPT = """\
Estimate the attributes of a new data science task.
 
Categories (choose exactly one):
{categories}
 
Similar past tasks, with their real attributes:
{examples}
 
New task:
"{description}"
 
Guidance:
- For the duration, use the typical value across the similar tasks that share
  the category you chose. Do not copy the single closest one: past tasks with
  the same wording vary widely in length, so their central value is a better
  estimate than any individual one.
- For the priority, judge urgency from the wording of the new task itself.
 
Answer with only this JSON object:
{{"category": "<one of the categories above>",
  "duration_min": <integer, minutes>,
  "priority": "<low, medium or high>"}}"""


def build_prompt(description: str, examples) -> str:
    """Assemble the prompt: category list, retrieved examples, new description.

    One line per example — enough signal, few tokens. The fixed instructions
    come first and the variable part last, so the provider's prompt cache can
    reuse the shared prefix.
    """
    example_lines = "\n".join(
        f'- "{row.description}" -> {row.category}, '
        f"{row.estimated_duration} minutes, {row.priority}"
        for row in examples.itertuples()
    )
    return PROMPT.format(
        categories="\n".join(f"- {c}" for c in CATEGORY_KEYS),
        examples=example_lines,
        description=description,
    )


def fallback(examples) -> dict:
    """What to answer when the model fails: the consensus of the neighbours.

    Median duration rather than mean, because one outlier among five would drag
    a mean but not a median.
    """
    return {
        "category": examples["category"].mode().iloc[0],
        "estimated_duration": int(examples["estimated_duration"].median()),
        "priority": examples["priority"].mode().iloc[0],
    }


def validate(reply: dict, examples) -> tuple[dict, list[str]]:
    """Check each field and replace the unusable ones.

    Repair is per field: if the model gets the category right and the duration
    wrong, the good part is kept. Returns the clean attributes and the list of
    fields that had to be repaired.
    """
    default = fallback(examples)
    repaired = []

    # Category: must be one of the eight. Anything else is unusable.
    category = reply.get("category")
    if category not in CATEGORY_TO_CONTEXT:
        category = default["category"]
        repaired.append("category")

    # Duration: a number inside its own category's range, widened by half.
    # A model answering 5 or 9000 minutes is not estimating, and that answer
    # must not reach the optimiser.
    low, high = CATEGORIES[category]["duration_range"]
    try:
        duration = int(float(reply["duration_min"]))
        assert low / 2 <= duration <= high * 1.5
    except (KeyError, TypeError, ValueError, AssertionError):
        duration = default["estimated_duration"]
        repaired.append("duration")

    # Priority: one of three labels, case-insensitive.
    priority = str(reply.get("priority", "")).strip().lower()
    if priority not in PRIORITY_LEVELS:
        priority = default["priority"]
        repaired.append("priority")

    return {
        "category": category,
        "context_group": CATEGORY_TO_CONTEXT[category],   # derived, never asked
        "estimated_duration": duration,
        "priority": priority,
        "priority_level": PRIORITY_LEVELS[priority],
    }, repaired


def infer_attributes(description: str, examples) -> dict:
    """Estimate the attributes of one task.

    'repaired' lists the fields the model got wrong: an empty list means the
    answer was used as given, three entries mean it was unusable and the
    neighbours answered instead.
    """
    reply = parse_json(ask(build_prompt(description, examples), system=SYSTEM)) or {}
    attributes, repaired = validate(reply, examples)
    return {**attributes, "repaired": repaired}