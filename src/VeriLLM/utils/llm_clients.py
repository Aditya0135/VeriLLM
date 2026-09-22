"""
Groq client construction and API-key rotation.

Free-tier quota is the binding constraint on this project (CLAUDE.md section
0), so every model gets a round-robin cycler over all available keys rather
than one client. Each model's cycle starts at a different offset, so three
models running back to back are never hitting the same key at the same moment.
"""

import itertools
import os

from dotenv import load_dotenv

from VeriLLM import logger

# How many GROQ_API_KEY<n> slots to look for in .env.
MAX_API_KEYS = 20


def load_api_keys(prefix="GROQ_API_KEY", max_keys=MAX_API_KEYS):
    """
    Collect `GROQ_API_KEY1..N` from the environment.

    Args:
        prefix (str): variable name prefix.
        max_keys (int): highest suffix to probe.

    Returns:
        list[str]: the keys that are actually set, in order.

    Raises:
        RuntimeError: if none are set. Failing here beats failing on row 1 of a
            3,351-row loop.
    """
    load_dotenv()

    keys = [
        os.getenv(f"{prefix}{index}")
        for index in range(1, max_keys + 1)
    ]

    keys = [key for key in keys if key]

    if not keys:
        raise RuntimeError(
            f"no {prefix}<n> values found in .env - annotation cannot run"
        )

    logger.info(f"loaded {len(keys)} Groq API keys")

    return keys


def create_llm_models(model_configs, api_keys, temperature=0, max_retries=2):
    """
    Build one `ChatGroq` client per (model, key) pair, as a cycler per model.

    Args:
        model_configs (dict): `{short_name: {"model": str, **extra_kwargs}}`,
            as `params.yaml` stores it. Extra kwargs are passed through, which
            is how `reasoning_effort` / `reasoning_format` reach the reasoning
            models.
        api_keys (list[str]): keys from `load_api_keys`.
        temperature (float): sampling temperature; 0 for reproducibility.
        max_retries (int): retries inside the client, below our own loop.

    Returns:
        dict: `{short_name: itertools.cycle([(key_id, client), ...])}`.

    Usage::

        models = create_llm_models(config.models, load_api_keys())
        key_id, client = next(models["gpt-oss-120b"])
    """
    from langchain_groq import ChatGroq

    models = {}

    for name, settings in model_configs.items():

        settings = dict(settings)
        model_id = settings.pop("model")

        clients = [
            (
                index + 1,
                ChatGroq(
                    api_key=key,
                    model=model_id,
                    temperature=temperature,
                    max_tokens=None,
                    timeout=None,
                    max_retries=max_retries,
                    **settings,
                ),
            )
            for index, key in enumerate(api_keys)
        ]

        # stagger each model's starting key so the models are never hitting
        # the same key at the same moment
        offset = len(models) % len(api_keys)
        clients = clients[offset:] + clients[:offset]

        models[name] = itertools.cycle(clients)

        logger.info(f"built {len(clients)} clients for {name} ({model_id})")

    return models
