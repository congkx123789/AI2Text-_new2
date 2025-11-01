"""
Observability setup (logging, tracing, metrics)
"""

from ai2text_common.observability.logging import setup_logging
from ai2text_common.observability.tracing import setup_tracing
from ai2text_common.observability.metrics import setup_metrics

__all__ = [
    "setup_logging",
    "setup_tracing",
    "setup_metrics",
]


