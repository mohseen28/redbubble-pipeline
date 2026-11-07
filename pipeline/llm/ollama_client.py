"""Ollama HTTP client (/api/chat, stream=false, format=<Pydantic JSON schema>)."""

from __future__ import annotations

import base64
import json
from typing import Any

import requests
from pydantic import ValidationError

from pipeline.config import LLMCfg
from pipeline.llm.base import LLMError, T
from pipeline.utils.logging import get_logger

log = get_logger("ollama")


class OllamaClient:
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:14b",
        *,
        fallback_model: str | None = None,
        temperature: float = 0.8,
        num_ctx: int = 4096,
        keep_alive: str = "10m",
        max_retries: int = 2,
        timeout: float = 600.0,
        session: requests.Session | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self._resolved = False
        self.calls = 0

    @classmethod
    def from_config(cls, cfg: LLMCfg, keep_alive: str | None = None) -> OllamaClient:
        return cls(cfg.host, cfg.text_model, fallback_model=cfg.fallback_model, temperature=cfg.temperature,
                   num_ctx=cfg.num_ctx, keep_alive=keep_alive if keep_alive is not None else cfg.keep_alive,
                   max_retries=cfg.max_json_retries, timeout=cfg.request_timeout_s)

    # ------------------------------------------------------------------ http
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self.session.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"cannot reach Ollama at {self.host}: {exc}. Is `ollama serve` running?") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama {path} returned HTTP {resp.status_code}: {resp.text[:300]}")
        data: dict[str, Any] = resp.json()
        return data

    def _get(self, path: str) -> dict[str, Any]:
        try:
            resp = self.session.get(f"{self.host}{path}", timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        data: dict[str, Any] = resp.json()
        return data

    def healthy(self) -> bool:
        try:
            self._get("/api/version")
            return True
        except LLMError:
            return False

    def local_models(self) -> list[str]:
        return [str(m.get("name", "")) for m in self._get("/api/tags").get("models", [])]

    def resolve_model(self) -> str:
        """Use text_model if pulled, else the fallback model (with a warning)."""
        if self._resolved:
            return self.model
        try:
            names = self.local_models()
        except LLMError:
            self._resolved = True       # let the chat call surface the real error
            return self.model

        def present(m: str) -> bool:
            return m in names or (":" not in m and f"{m}:latest" in names)

        if not present(self.model):
            if self.fallback_model and present(self.fallback_model):
                log.warning("model %s not pulled; using fallback %s", self.model, self.fallback_model)
                self.model = self.fallback_model
            else:
                raise LLMError(f"model {self.model!r} is not pulled in Ollama (have: {names}). "
                               f"Run: ollama pull {self.model}")
        self._resolved = True
        return self.model

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict[str, Any]], schema: dict[str, Any] | None = None, *,
             model: str | None = None, temperature: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": model or self.resolve_model(),
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_ctx": self.num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        self.calls += 1
        data = self._post("/api/chat", payload)
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMError(f"unexpected Ollama response: {json.dumps(data)[:300]}")
        return content

    def complete_json(self, system: str, user: str, schema: type[T], *, model: str | None = None,
                      images: list[bytes] | None = None, temperature: float | None = None) -> T:
        user_msg: dict[str, Any] = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, user_msg]
        json_schema = schema.model_json_schema()
        last_error = ""
        for attempt in range(self.max_retries + 1):
            content = self.chat(messages, json_schema, model=model, temperature=temperature)
            try:
                return schema.model_validate_json(_strip_fences(content))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:500]
                log.warning("invalid JSON from LLM (attempt %d/%d): %s", attempt + 1,
                            self.max_retries + 1, last_error.splitlines()[0] if last_error else "")
                messages = [*messages, {"role": "assistant", "content": content},
                            {"role": "user", "content": "Your reply did not match the JSON schema: "
                             f"{last_error}\nReply again with ONLY valid JSON matching the schema."}]
        raise LLMError(f"no schema-valid JSON after {self.max_retries + 1} attempts: {last_error}")

    def unload(self, model: str | None = None) -> None:
        """keep_alive=0 makes Ollama free the model's VRAM immediately."""
        self._post("/api/generate", {"model": model or self.model, "keep_alive": 0})


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()
