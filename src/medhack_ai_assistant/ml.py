from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import httpx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ModelConfig, VLLMConfig


@dataclass(frozen=True)
class ValidationResult:
    threshold: float
    f1: float


MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]
    usage: LLMUsage
    model: str
    latency_seconds: float


class VLLMClientError(RuntimeError):
    """Raised when the vLLM endpoint cannot serve a request safely."""


class VLLMResponseFormatError(VLLMClientError):
    """Raised when vLLM returns a syntactically valid but unusable response."""


class AsyncVLLMClient:
    """Async OpenAI-compatible client for the remote vLLM endpoint.

    The remote model is expected to be exposed through an SSH tunnel, usually at
    http://127.0.0.1:8000/v1/chat/completions.
    """

    _RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        config: VLLMConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or VLLMConfig()
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "AsyncVLLMClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def chat(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)

        raw, latency_seconds = await self._post_with_retries(payload)
        content = _extract_message_content(raw)
        usage = _parse_usage(raw.get("usage"))

        return LLMResponse(
            content=content,
            raw=raw,
            usage=usage,
            model=str(raw.get("model") or self.config.model),
            latency_seconds=latency_seconds,
        )

    async def chat_json(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self.chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_payload=extra_payload,
        )
        return parse_json_response(response.content)

    async def _post_with_retries(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        client = await self._ensure_client()
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        start = time.perf_counter()

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.post(
                    self.config.chat_completions_url,
                    headers=headers,
                    json=payload,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                await self._sleep_before_retry(attempt)
                continue
            except httpx.HTTPError as exc:
                raise VLLMClientError(
                    "vLLM request failed before receiving a valid HTTP response. "
                    "Check the SSH tunnel and vLLM endpoint."
                ) from exc

            if response.status_code in self._RETRY_STATUS_CODES:
                last_error = VLLMClientError(
                    f"vLLM returned transient HTTP {response.status_code}"
                )
                if attempt >= self.config.max_retries:
                    break
                await self._sleep_before_retry(attempt)
                continue

            if response.is_error:
                raise VLLMClientError(
                    f"vLLM returned HTTP {response.status_code}. "
                    "Check model name, tunnel target, and server logs."
                )

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise VLLMResponseFormatError("vLLM returned non-JSON response") from exc

            if not isinstance(payload, dict):
                raise VLLMResponseFormatError("vLLM response root must be a JSON object")

            return payload, time.perf_counter() - start

        raise VLLMClientError(
            "vLLM endpoint is unavailable. Check that the SSH tunnel is running "
            f"and reachable at {self.config.chat_completions_url}."
        ) from last_error

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self.config.timeout_seconds)
            self._client = httpx.AsyncClient(timeout=timeout)
            self._owns_client = True
        return self._client

    async def _sleep_before_retry(self, attempt: int) -> None:
        await asyncio.sleep(min(0.5 * (2**attempt), 4.0))


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse strict or fenced JSON returned by an instruction-tuned model."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_markdown_fence(cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise VLLMResponseFormatError("vLLM response content is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise VLLMResponseFormatError("vLLM JSON response must be an object")
    return payload


def _strip_markdown_fence(content: str) -> str:
    lines = content.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return content


def _message_to_dict(message: LLMMessage | dict[str, str]) -> dict[str, str]:
    if isinstance(message, LLMMessage):
        role = message.role
        content = message.content
    else:
        role = message.get("role", "")
        content = message.get("content", "")

    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Unsupported LLM message role: {role}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM message content must be a non-empty string")

    return {"role": role, "content": content}


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VLLMResponseFormatError("vLLM response does not contain choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise VLLMResponseFormatError("vLLM choice must be a JSON object")

    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = first_choice.get("text")

    if not isinstance(content, str) or not content.strip():
        raise VLLMResponseFormatError("vLLM response does not contain message content")

    return content.strip()


def _parse_usage(value: Any) -> LLMUsage:
    if not isinstance(value, dict):
        return LLMUsage()
    return LLMUsage(
        prompt_tokens=_optional_int(value.get("prompt_tokens")),
        completion_tokens=_optional_int(value.get("completion_tokens")),
        total_tokens=_optional_int(value.get("total_tokens")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_model(config: ModelConfig) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=(config.ngram_min, config.ngram_max),
                min_df=config.min_df,
                lowercase=True,
            ),
        ),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=config.random_state,
                max_iter=1000,
                tol=1e-3,
            ),
        ),
    ])


def find_best_threshold(
    y_true: pd.Series,
    positive_proba: np.ndarray,
    config: ModelConfig,
) -> ValidationResult:
    thresholds = np.linspace(
        config.threshold_min,
        config.threshold_max,
        config.threshold_count,
    )
    scores = [
        f1_score(y_true, positive_proba >= threshold)
        for threshold in thresholds
    ]

    best_idx = int(np.argmax(scores))
    return ValidationResult(
        threshold=float(thresholds[best_idx]),
        f1=float(scores[best_idx]),
    )


def train_and_validate(
    x: pd.Series,
    y: pd.Series,
    config: ModelConfig,
) -> ValidationResult:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = build_model(config)
    model.fit(x_train, y_train)
    valid_proba = model.predict_proba(x_valid)[:, 1]

    return find_best_threshold(y_valid, valid_proba, config)


def train_final_model(
    x: pd.Series,
    y: pd.Series,
    config: ModelConfig,
) -> Pipeline:
    model = build_model(config)
    model.fit(x, y)
    return model


def predict_with_threshold(
    model: Pipeline,
    x: pd.Series,
    threshold: float,
) -> pd.Series:
    positive_proba = model.predict_proba(x)[:, 1]
    return pd.Series(positive_proba >= threshold, index=x.index)
