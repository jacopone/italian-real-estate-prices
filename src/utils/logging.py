"""Loguru logging configuration.

This module provides a consistent logging setup across the entire pipeline.
Loguru is used instead of stdlib logging for its cleaner API and better defaults.

Usage:
    from src.utils.logging import setup_logging, get_logger

    # At application start
    setup_logging(level="INFO", log_file="outputs/pipeline.log")

    # In modules
    logger = get_logger(__name__)
    logger.info("Processing data", rows=1000)
"""

import sys
from pathlib import Path
from typing import Literal

from loguru import logger

# Type alias for log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    colorize: bool = True,
    json_format: bool = False,
) -> None:
    """Configure Loguru logging for the pipeline.

    Sets up console and optional file logging with consistent formatting.

    Args:
        level: Minimum log level to display.
        log_file: Path to log file. If None, logs only to console.
        rotation: When to rotate log files (size or time).
        retention: How long to keep rotated logs.
        colorize: Whether to colorize console output.
        json_format: If True, output structured JSON logs.

    Example:
        >>> setup_logging(level="DEBUG", log_file="outputs/run.log")
        >>> logger.info("Pipeline started", config="default.yaml")
    """
    # Remove default handler
    logger.remove()

    # Console format with context
    if json_format:
        console_format = "{message}"
        serialize = True
    else:
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        serialize = False

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=colorize,
        serialize=serialize,
    )

    # Add file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            str(log_path),
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=json_format,
        )

        logger.info(f"Logging to file: {log_path}")


def get_logger(name: str):
    """Get a contextualized logger for a module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Bound logger with module context.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Loading data")
    """
    return logger.bind(module=name)


# =============================================================================
# LOGGING CONTEXT MANAGERS
# =============================================================================


class LogContext:
    """Context manager for adding temporary context to logs.

    Example:
        >>> with LogContext(task="data_loading", source="OMI"):
        ...     logger.info("Starting")  # Includes task and source
        >>> logger.info("Done")  # No longer includes context
    """

    def __init__(self, **kwargs):
        """Initialize with context variables."""
        self.context = kwargs
        self._token = None

    def __enter__(self):
        """Add context on entry."""
        self._token = logger.contextualize(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove context on exit."""
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)
        return False


def log_dataframe_info(df, name: str = "DataFrame") -> None:
    """Log DataFrame summary information.

    Useful for debugging data loading and transformation steps.

    Args:
        df: pandas DataFrame to summarize.
        name: Name to identify the DataFrame in logs.
    """
    logger.info(
        f"{name} summary",
        rows=len(df),
        columns=len(df.columns),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1_000_000, 2),
        column_list=list(df.columns[:10]),  # First 10 columns
    )


def log_model_metrics(
    model_name: str,
    r2: float,
    rmse: float | None = None,
    mae: float | None = None,
    **extra_metrics,
) -> None:
    """Log model performance metrics in a consistent format.

    Args:
        model_name: Name of the model.
        r2: R-squared score.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        **extra_metrics: Additional metrics to log.
    """
    metrics = {"model": model_name, "r2": round(r2, 4)}
    if rmse is not None:
        metrics["rmse"] = round(rmse, 4)
    if mae is not None:
        metrics["mae"] = round(mae, 4)
    metrics.update(extra_metrics)

    logger.info("Model evaluation", **metrics)


# =============================================================================
# PROGRESS LOGGING
# =============================================================================


class ProgressLogger:
    """Simple progress logger for long-running operations.

    Example:
        >>> progress = ProgressLogger(total=1000, description="Processing")
        >>> for item in items:
        ...     process(item)
        ...     progress.update()
        >>> progress.finish()
    """

    def __init__(
        self,
        total: int,
        description: str = "Progress",
        log_interval: int = 10,
    ):
        """Initialize progress logger.

        Args:
            total: Total number of items.
            description: Description for log messages.
            log_interval: Percentage interval for logging (default: every 10%).
        """
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_pct = 0

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
        pct = int(100 * self.current / self.total)

        # Log at intervals
        if pct >= self.last_logged_pct + self.log_interval:
            self.last_logged_pct = (pct // self.log_interval) * self.log_interval
            logger.info(
                f"{self.description}: {pct}%",
                current=self.current,
                total=self.total,
            )

    def finish(self) -> None:
        """Log completion."""
        logger.info(f"{self.description}: Complete", total=self.total)


# Initialize default console logging on import
setup_logging(level="INFO")
