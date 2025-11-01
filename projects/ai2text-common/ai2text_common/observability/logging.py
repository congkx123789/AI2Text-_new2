"""
Logging setup with structured logging
"""

import logging
import sys
from typing import Optional


def setup_logging(
    service_name: str,
    level: str = "INFO",
    format_type: str = "json",
) -> logging.Logger:
    """
    Setup structured logging for a service

    Args:
        service_name: Name of the service
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("json" or "text")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if format_type == "json":
        # JSON logging (optional dependency)
        try:
            import json_log_formatter
            formatter = json_log_formatter.JSONFormatter()
        except ImportError:
            # Fallback to text if json_log_formatter not installed
            formatter = logging.Formatter(
                f"%(asctime)s [%(levelname)s] {service_name}: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
    else:
        formatter = logging.Formatter(
            f"%(asctime)s [%(levelname)s] {service_name}: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

