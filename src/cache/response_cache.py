"""TTL-aware response cache for all 18 use cases."""
from __future__ import annotations
import hashlib
import json
import time
from typing import Optional, Protocol, runtime_checkable
from pathlib import Path

DEFAULT_TTLS = {
    "triage": 300,
    "compliance_qa": 3600,
    "incident": 60,
    "reconcile": 1800,
    "outreach": 3600,
    "report": 900,
    "review_pr": 600,
    "exception": 300,
    "rfp": 7200,
    "clinical_decision": 0,
    "drug_interaction": 86400,
    "literature": 86400,
    "patient_risk": 0,
    "healthcare_gaps": 3600,
    "genomic_risk": 0,
    "mental_health": 0,
    "clinical_trials": 1800,
    "stock": 300,
}


@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl_seconds: int = 3600): ...
    def delete(self, key: str): ...
    def clear(self): ...


class InMemoryCache:
    def __init__(self, maxsize: int = 256):
        self._store: dict[str, tuple[str, float]] = {}
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires = entry
        if time.time() > expires:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int = 3600):
        if ttl_seconds == 0:
            return
        if len(self._store) >= self._maxsize:
            oldest = min(self._store.items(), key=lambda x: x[1][1])[0]
            del self._store[oldest]
        self._store[key] = (value, time.time() + ttl_seconds)

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()


class DiskCache:
    def __init__(self, directory: str = "agent_cache"):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / (hashlib.md5(key.encode()).hexdigest() + ".json")

    def get(self, key: str) -> Optional[str]:
        p = self._path(key)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        if time.time() > data["expires"]:
            p.unlink(missing_ok=True)
            return None
        return data["value"]

    def set(self, key: str, value: str, ttl_seconds: int = 3600):
        if ttl_seconds == 0:
            return
        self._path(key).write_text(json.dumps({"value": value, "expires": time.time() + ttl_seconds}))

    def delete(self, key: str):
        self._path(key).unlink(missing_ok=True)

    def clear(self):
        for f in self._dir.glob("*.json"):
            f.unlink(missing_ok=True)


class ResponseCache:
    def __init__(self, backend: Optional[CacheBackend] = None, default_ttl: int = 3600, enabled: bool = True):
        self._backend = backend or InMemoryCache()
        self._default_ttl = default_ttl
        self.enabled = enabled

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def cache_key(self, use_case: str, input_text: str) -> str:
        return f"{use_case}:{self._hash(input_text)}"

    def get_cached(self, use_case: str, input_text: str) -> Optional[str]:
        if not self.enabled:
            return None
        return self._backend.get(self.cache_key(use_case, input_text))

    def cache_result(self, use_case: str, input_text: str, result_json: str, ttl: Optional[int] = None):
        if not self.enabled:
            return
        ttl = ttl if ttl is not None else DEFAULT_TTLS.get(use_case, self._default_ttl)
        self._backend.set(self.cache_key(use_case, input_text), result_json, ttl)
