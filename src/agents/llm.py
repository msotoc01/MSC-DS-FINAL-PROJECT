import hashlib
import json
import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel

import config

load_dotenv()


TOKEN_USAGE = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}

_model = None


def get_model() -> OpenAIServerModel:
    """The shared model instance, created once on first use."""
    global _model
    if _model is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Copy .env.example to .env and add your key."
            )
        _model = OpenAIServerModel(model_id=config.LLM_MODEL_ID, api_key=api_key)
    return _model