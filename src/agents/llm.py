import hashlib
import json
import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel

import config

load_dotenv()

CALLS = {"api": 0, "cached": 0}
_model = None


def get_model() -> OpenAIServerModel:
    """The shared model instance, created once on first use. This
    module can be imported without an API key."""
    global _model
    if _model is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not found. Add it to .env at the repo root.")
        _model = OpenAIServerModel(model_id=config.LLM_MODEL_ID, api_key=key)
    return _model


def ask(prompt: str, system: str = "", temperature: float = 0.0,
        use_cache: bool = True) -> str:
    """Send a prompt, return the text reply.

    temperature defaults to 0: attribute inference must be reproducible.

    One file per call, not one big JSON: an interrupted run cannot corrupt
    earlier entries.
    """
    payload = json.dumps({"prompt": prompt, "system": system,
                          "model": config.LLM_MODEL_ID, "temperature": temperature},
                         sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()[:32]
    path = config.LLM_CACHE_DIR / f"{key}.json"

    if use_cache and path.exists():
        CALLS["cached"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    text = get_model().generate(messages, temperature=temperature).content or ""
    CALLS["api"] += 1

    if use_cache:
        config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompt": prompt, "system": system, "response": text},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return text


def parse_json(raw: str):
    """Pull a JSON object out of a reply, or return None.

    Models wrap JSON in ``` fences or add a sentence around it. 
    One bad reply on one task of 600 must not kill an evaluation run.
    """
    if not raw:
        return None
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None