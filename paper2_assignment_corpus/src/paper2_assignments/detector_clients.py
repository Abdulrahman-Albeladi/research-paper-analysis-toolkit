from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import requests


class DetectorAPIError(RuntimeError):
    """Raised when a detector request fails or returns an unsupported response."""


@dataclass(frozen=True)
class DetectorResult:
    score_percent: float
    label: str
    raw: dict[str, Any]


def _normalize_percent(value: Any) -> float:
    if value is None:
        return np.nan
    numeric = float(value)
    if numeric <= 1.000001:
        numeric *= 100.0
    return round(float(np.clip(numeric, 0.0, 100.0)), 6)


def _label_from_score(score: float, supplied: Any = None) -> str:
    if supplied is not None and str(supplied).strip():
        return str(supplied).strip()
    return "Likely AI" if score >= 50 else "Likely Human"


class DetectorClients:
    """HTTP clients matching the commercial endpoints used in the study notebooks.

    Credentials are read from environment variables and are never included in
    return values, logs, or exception messages.
    """

    def __init__(
        self,
        *,
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_wait_seconds: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.session = session or requests.Session()

    @staticmethod
    def required_key(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise DetectorAPIError(f"Required environment variable is not set: {name}")
        return value

    def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
                if 200 <= response.status_code < 300:
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise DetectorAPIError("Detector returned a non-object JSON response.")
                    return payload
                last_error = DetectorAPIError(
                    f"Detector HTTP error {response.status_code}; response body omitted."
                )
            except Exception as exc:  # network and JSON errors
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_wait_seconds)
        raise DetectorAPIError(str(last_error) if last_error else "Detector request failed.")

    def gptzero(self, text: str) -> DetectorResult:
        api_key = self.required_key("GPTZERO_API_KEY")
        payload = self._request_json(
            "POST",
            "https://api.gptzero.me/v2/predict/text",
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={"document": text},
        )
        document: dict[str, Any]
        if isinstance(payload.get("documents"), list) and payload["documents"]:
            document = payload["documents"][0]
        else:
            document = payload
        probabilities = document.get("class_probabilities") or {}
        score = _normalize_percent(probabilities.get("ai"))
        label = _label_from_score(
            score,
            document.get("predicted_class") or document.get("document_classification"),
        )
        return DetectorResult(score, label, payload)

    def pangram(self, text: str) -> DetectorResult:
        api_key = self.required_key("PANGRAM_API_KEY")
        payload = self._request_json(
            "POST",
            "https://text.api.pangram.com/v3",
            headers={"Content-Type": "application/json", "x-api-key": api_key},
            json={"text": text, "public_dashboard_link": False},
        )
        score = _normalize_percent(payload.get("fraction_ai"))
        label = _label_from_score(
            score, payload.get("prediction_short") or payload.get("prediction")
        )
        return DetectorResult(score, label, payload)

    def sapling(self, text: str) -> DetectorResult:
        api_key = self.required_key("SAPLING_API_KEY")
        payload = self._request_json(
            "POST",
            "https://api.sapling.ai/api/v1/aidetect",
            json={"key": api_key, "text": text},
        )
        score = _normalize_percent(payload.get("score"))
        return DetectorResult(score, _label_from_score(score), payload)

    def isgen(self, text: str, language: str) -> DetectorResult:
        api_key = self.required_key("ISGEN_RAPIDAPI_KEY")
        lang = "ar" if language == "Arabic" else "en"
        payload = self._request_json(
            "POST",
            "https://ai-detection4.p.rapidapi.com/v1/ai-detection-rapid-api",
            headers={
                "x-rapidapi-key": api_key,
                "x-rapidapi-host": "ai-detection4.p.rapidapi.com",
                "Content-Type": "application/json",
            },
            json={"text": text, "lang": lang},
        )
        score = _normalize_percent(payload.get("ai_score"))
        return DetectorResult(score, _label_from_score(score), payload)

    def get(self, tool: str) -> Callable[..., DetectorResult]:
        clients: dict[str, Callable[..., DetectorResult]] = {
            "gptzero": self.gptzero,
            "pangram": self.pangram,
            "sapling": self.sapling,
            "isgen": self.isgen,
        }
        try:
            return clients[tool]
        except KeyError as exc:
            raise ValueError(f"Unsupported detector: {tool}") from exc
