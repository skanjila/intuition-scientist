"""Observability and tracing package."""
from src.telemetry.otel_tracer import get_tracer, record_use_case_metric
__all__ = ["get_tracer", "record_use_case_metric"]
