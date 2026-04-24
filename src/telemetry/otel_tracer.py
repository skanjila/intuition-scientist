"""OpenTelemetry tracer — graceful no-op when OTEL not installed."""
from __future__ import annotations
import os
import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _OTEL_AVAILABLE = True
except ImportError:
    pass


class _NoOpSpan:
    def set_attribute(self, k, v): pass
    def record_exception(self, exc): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass


class _NoOpTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **_) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()


_tracer_instance: Any = None


def get_tracer(service_name: str = "business-agent-platform") -> Any:
    global _tracer_instance
    if _tracer_instance is not None:
        return _tracer_instance
    if _OTEL_AVAILABLE:
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                provider = TracerProvider()
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                trace.set_tracer_provider(provider)
                _tracer_instance = trace.get_tracer(service_name)
                logger.info("OTEL tracing enabled → %s", endpoint)
                return _tracer_instance
            except Exception as exc:
                logger.warning("OTEL setup failed: %s — using no-op tracer", exc)
    _tracer_instance = _NoOpTracer()
    return _tracer_instance


def record_use_case_metric(use_case: str, latency_ms: float, confidence: float, escalated: bool) -> None:
    logger.debug("metric use_case=%s latency_ms=%.1f confidence=%.2f escalated=%s", use_case, latency_ms, confidence, escalated)
