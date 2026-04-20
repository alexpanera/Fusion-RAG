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
