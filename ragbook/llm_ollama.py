from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

from ragbook.utils import LOGGER

PREFERRED_MODELS = [
    "qwen2.5:7b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:3b",
    "qwen:0.5b",
    "qwen2.5:0.5b",
]


def _request_timeout_seconds() -> int:
    raw = os.getenv("OLLAMA_TIMEOUT_SEC", "600")
    try:
        return max(1, int(raw))
    except ValueError:
        LOGGER.warning("Invalid OLLAMA_TIMEOUT_SEC=%r. Falling back to 600.", raw)
        return 600


@dataclass
class OllamaClient:
    host: str
    model: str

    @classmethod
    def create(
        cls,
        host: str | None = None,
        model_override: str | None = None,
    ) -> "OllamaClient":
        ollama_host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        chosen = model_override or os.getenv("OLLAMA_MODEL")
        if not chosen:
            available = _list_local_models(ollama_host)
            chosen = _select_preferred_model(available)
            if not chosen:
                raise RuntimeError(
                    "No suitable Ollama model found locally. Pull one of: "
                    + ", ".join(PREFERRED_MODELS)
                )
        LOGGER.info("Using Ollama model: %s", chosen)
        return cls(host=ollama_host, model=chosen)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Send a prompt via the /api/chat endpoint (system + user split).

        The chat endpoint works significantly better than /api/generate for
        instruction-tuned models because the system role and user role are kept
        separate, matching how the model was fine-tuned.

        The prompt is expected to follow the convention used by build_answer_prompt:
        an instruction block followed by 'Question:' and 'Context:' sections.
        We split it at 'Question:' so instructions become the system message and
        the question + context become the user message.
        """
        timeout_sec = _request_timeout_seconds()

        split_marker = "\nQuestion:\n"
        if split_marker in prompt:
            system_part, user_part = prompt.split(split_marker, 1)
            system_msg = system_part.strip()
            # Strip the "Final answer:" generation cue — it was a hint for the
            # raw /api/generate endpoint but confuses the chat API (the model
            # sees it as part of the user message rather than a generation start).
            user_text = ("Question:\n" + user_part).rstrip()
            if user_text.endswith("Final answer:"):
                user_text = user_text[: -len("Final answer:")].rstrip()
            user_msg = user_text
        else:
            system_msg = "You are a helpful assistant."
            user_msg = prompt.strip()

        def _request(model: str) -> requests.Response:
            url = f"{self.host}/api/chat"
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            }
            return requests.post(url, json=payload, timeout=timeout_sec)

        def _missing_model(resp: requests.Response) -> bool:
            if resp.status_code != 404:
                return False
            try:
                msg = str(resp.json().get("error", "")).lower()
            except Exception:
                msg = (resp.text or "").lower()
            return "model" in msg and ("not found" in msg or "no such" in msg)

        try:
            resp = _request(self.model)
            if _missing_model(resp):
                available = _list_local_models(self.host)
                fallback = _select_preferred_model([m for m in available if m != self.model])
                if fallback:
                    LOGGER.warning(
                        "Model '%s' not found in Ollama. Retrying with '%s'.",
                        self.model,
                        fallback,
                    )
                    self.model = fallback
                    resp = _request(self.model)
            resp.raise_for_status()
            data = resp.json()
            # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
            return (data.get("message", {}).get("content") or "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e


def _parse_param_billions(model: str) -> float:
    """Extract parameter count in billions from a model name string.

    Examples: 'qwen2.5:7b' → 7.0, 'qwen2.5:0.5b' → 0.5, 'llama3.1:8b' → 8.0
    Returns 7.0 as a safe default when the size cannot be parsed.
    """
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model, re.I)
    return float(m.group(1)) if m else 7.0


def _model_on_gpu(host: str, model: str) -> bool:
    """Return True if Ollama is running the model with VRAM (i.e. on a GPU).

    Queries /api/ps which lists currently-loaded models with their VRAM usage.
    Falls back to False (assume CPU) on any error.
    """
    try:
        resp = requests.get(f"{host}/api/ps", timeout=5)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            # entry["name"] may be "qwen2.5:7b" while model is "qwen2.5:7b"
            if model.split(":")[0] in entry.get("name", ""):
                return int(entry.get("size_vram", 0)) > 0
    except Exception:
        pass
    return False


def suggest_rag_settings(host: str, model: str) -> tuple[int, int]:
    """Return (top_k, max_context_chars) tuned for the given model and hardware.

    Priority: explicit env vars > auto-detection.
    This is called before the first query so that sensible defaults are already
    in place when the user has not set RAG_TOP_K / RAG_MAX_CONTEXT_CHARS.

    Detection logic:
    - Parse parameter count from the model name (e.g. '7b' → 7.0 B)
    - Check Ollama /api/ps to see if the model is running on GPU (VRAM > 0)
    - Look up a settings table: (size bucket, gpu) → (top_k, context_chars)
    """
    # Respect explicit env-var overrides — don't touch what the user set.
    env_top_k = os.getenv("RAG_TOP_K")
    env_chars = os.getenv("RAG_MAX_CONTEXT_CHARS")
    if env_top_k and env_chars:
        return int(env_top_k), int(env_chars)

    params = _parse_param_billions(model)
    on_gpu = _model_on_gpu(host, model)

    # Lookup table: each row is (max_params, on_gpu, top_k, context_chars)
    # Rows are checked in order; first match wins.
    TABLE = [
        # Small models are fast on both CPU and GPU
        (3.0,  False, 2, 3_000),
        (3.0,  True,  6, 8_000),
        # Mid-size (7-8 B): comfortable on GPU, tight on CPU
        (8.0,  False, 2, 3_000),
        (8.0,  True,  6, 8_000),
        # Large (13-14 B)
        (14.0, False, 2, 2_000),
        (14.0, True,  8, 12_000),
        # Very large (30 B+): only realistic on GPU
        (float("inf"), False, 2, 2_000),
        (float("inf"), True,  8, 16_000),
    ]

    for max_p, gpu, top_k, chars in TABLE:
        if params <= max_p and on_gpu == gpu:
            break

    # Apply any partial env-var override
    if env_top_k:
        top_k = int(env_top_k)
    if env_chars:
        chars = int(env_chars)

    LOGGER.info(
        "Auto-settings: model=%s params=%.1fB gpu=%s → top_k=%d context_chars=%d",
        model, params, on_gpu, top_k, chars,
    )
    return top_k, chars


def _list_models_from_cli() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return []

    models: list[str] = []
    for ln in lines[1:]:
        name = ln.split()[0]
        if name and name.lower() != "name":
            models.append(name)
    return models


def _list_models_from_api(host: str) -> list[str]:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except Exception:
                                    return []


def _list_local_models(host: str) -> list[str]:
    cli_models = _list_models_from_cli()
    if cli_models:
        return cli_models
    return _list_models_from_api(host)


def _select_preferred_model(models: list[str]) -> str | None:
    model_set = set(models)
    for preferred in PREFERRED_MODELS:
        if preferred in model_set:
            return preferred
    return models[0] if models else None
