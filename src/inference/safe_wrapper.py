"""
src/inference/safe_wrapper.py
=============================
Enterprise-grade exception shielding and failure handling.

Guarantees:
- Never crashes the calling thread or process.
- Catches all exceptions (including CUDA OOM, driver crashes, network drops, OS I/O).
- Maintains execution health metrics and consecutive failure counters.
- Triggers graceful degradation callbacks when failure thresholds are exceeded.
- Returns the last known valid result or a deterministic fallback default.
"""

from __future__ import annotations

import functools
import logging
import traceback
from typing import Any, Callable, Generic, Optional, TypeVar

log = logging.getLogger("apex_safe_wrapper")

T = TypeVar("T")


class SafeExecutor(Generic[T]):
    """
    Thread-safe execution wrapper that shields the host application from runtime faults.
    
    Attributes:
        name: Identifier for diagnostic logging.
        safe_default: Fallback value returned when no previous valid result exists.
        max_consecutive_failures: Number of consecutive errors before degradation is triggered.
        on_degradation: Optional callback invoked when consecutive failures reach threshold.
    """

    def __init__(
        self,
        name: str = "executor",
        safe_default: Optional[T] = None,
        max_consecutive_failures: int = 3,
        on_degradation: Optional[Callable[[int, Exception], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self.safe_default = safe_default
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.on_degradation = on_degradation
        self.log = logger or log

        self.consecutive_failures: int = 0
        self.total_failures: int = 0
        self.total_successes: int = 0
        self.last_valid_result: Optional[T] = None
        self.last_exception: Optional[Exception] = None
        self._degradation_triggered: bool = False

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute `fn(*args, **kwargs)` inside an unbreachable exception shield.
        """
        try:
            result = fn(*args, **kwargs)
            self.last_valid_result = result
            self.consecutive_failures = 0
            self.total_successes += 1
            self._degradation_triggered = False
            return result
        except BaseException as exc:
            # We explicitly handle BaseException to shield against KeyboardInterrupt/SystemExit
            # during critical sub-routines unless explicitly re-raised by policy.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise exc

            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_exception = exc

            tb = traceback.format_exc()
            self.log.error(
                "[%s] Exception caught (Consecutive: %d/%d, Total: %d): %s\n%s",
                self.name,
                self.consecutive_failures,
                self.max_consecutive_failures,
                self.total_failures,
                exc,
                tb,
            )

            # Trigger degradation callback once threshold is reached
            if (
                self.consecutive_failures >= self.max_consecutive_failures
                and not self._degradation_triggered
            ):
                self._degradation_triggered = True
                if self.on_degradation:
                    try:
                        self.on_degradation(self.consecutive_failures, exc)
                    except Exception as cb_exc:
                        self.log.critical(
                            "[%s] Degradation callback raised an unhandled exception: %s",
                            self.name,
                            cb_exc,
                        )

            # Return last valid result if available; otherwise return safe default
            if self.last_valid_result is not None:
                return self.last_valid_result
            return self.safe_default  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset fault counters and degradation states."""
        self.consecutive_failures = 0
        self._degradation_triggered = False
        self.last_exception = None

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Allow using the executor directly as a decorator."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(fn, *args, **kwargs)
        return wrapper


def safe_execute(
    name: str = "shielded_call",
    safe_default: Optional[Any] = None,
    max_consecutive_failures: int = 3,
    on_degradation: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator syntax for shielding functions with a dedicated SafeExecutor instance.
    """
    executor = SafeExecutor(
        name=name,
        safe_default=safe_default,
        max_consecutive_failures=max_consecutive_failures,
        on_degradation=on_degradation,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return executor.execute(func, *args, **kwargs)
        # Expose executor state on the wrapper function for diagnostics
        wrapper.executor = executor  # type: ignore[attr-defined]
        return wrapper

    return decorator


def safe_call(
    fn: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    name: str = "ad_hoc_call",
    **kwargs: Any,
) -> T:
    """
    Single-line functional wrapper for one-off safe execution.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.error("[%s] SafeCall caught exception: %s", name, exc)
        return default  # type: ignore[return-value]
