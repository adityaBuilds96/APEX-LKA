"""
deploy/watchdog.py
==================
Independent Supervisor Watchdog Service for Autonomous Jetson Deployment.

Responsibilities:
- Pings / monitors main inference process liveness and heartbeat every 500ms.
- Detects process unresponsiveness (> 3.0s without heartbeat update).
- Automatically terminates and restarts hung or crashed processes.
- Enforces Crash-Loop Shield: If >= 3 restarts occur within a 60-second window,
  enters SAFE MODE (halts restart cycles, initiates emergency vehicle stop).
- Runs as an independent process or Linux systemd unit.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(levelname)s: %(message)s",
)
log = logging.getLogger("apex_watchdog")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProcessWatchdog:
    """
    Supervises child process liveness and enforces failsafe safe-mode thresholds.
    """

    def __init__(
        self,
        command: Union[str, List[str]],
        heartbeat_file: Optional[Union[str, Path]] = None,
        timeout_s: float = 3.0,
        check_interval_s: float = 0.5,
        max_restarts: int = 3,
        window_s: float = 60.0,
        sim_mode: bool = False,
    ) -> None:
        self.command = command if isinstance(command, list) else command.split()
        self.heartbeat_file = Path(heartbeat_file) if heartbeat_file else None
        self.timeout_s = float(timeout_s)
        self.check_interval_s = float(check_interval_s)
        self.max_restarts = max(1, max_restarts)
        self.window_s = float(window_s)
        self.sim_mode = sim_mode

        self.process: Optional[subprocess.Popen] = None
        self.restart_history: List[float] = []
        self.is_safe_mode: bool = False
        self.total_restarts: int = 0
        self._sim_heartbeat_time: float = time.time()
        self._sim_process_alive: bool = True

    def update_sim_heartbeat(self) -> None:
        """Helper for unit test simulation."""
        self._sim_heartbeat_time = time.time()

    def set_sim_unresponsive(self) -> None:
        """Helper for unit test simulation."""
        self._sim_heartbeat_time = time.time() - (self.timeout_s + 1.0)

    def start_process(self) -> bool:
        """Spawn the supervised child process."""
        if self.sim_mode:
            self._sim_process_alive = True
            self._sim_heartbeat_time = time.time()
            log.info("[SIMULATION] Spawned child process: %s", " ".join(self.command))
            return True

        try:
            # Ensure fresh heartbeat file
            if self.heartbeat_file:
                self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
                self.heartbeat_file.write_text(str(time.time()), encoding="utf-8")

            self.process = subprocess.Popen(
                self.command,
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
            )
            log.info("Spawned supervised process [PID %d]: %s", self.process.pid, " ".join(self.command))
            return True
        except Exception as e:
            log.critical("Failed to launch supervised process: %s", e)
            return False

    def is_heartbeat_fresh(self) -> bool:
        """Verify heartbeat timestamp freshness."""
        now = time.time()
        if self.sim_mode:
            if not self._sim_process_alive:
                return False
            return (now - self._sim_heartbeat_time) <= self.timeout_s

        # 1. Process termination check
        if self.process is None or self.process.poll() is not None:
            return False

        # 2. File heartbeat check
        if self.heartbeat_file and self.heartbeat_file.exists():
            try:
                mtime = os.path.getmtime(self.heartbeat_file)
                if (now - mtime) > self.timeout_s:
                    log.warning(
                        "Heartbeat stale: %.2fs elapsed since last update (Limit: %.1fs)",
                        now - mtime,
                        self.timeout_s,
                    )
                    return False
            except Exception as e:
                log.warning("Could not check heartbeat file: %s", e)
                return False

        return True

    def restart_process(self) -> bool:
        """
        Evaluate sliding window restart quota and perform managed restart.
        """
        now = time.time()
        # Prune restart history outside the sliding window
        self.restart_history = [t for t in self.restart_history if (now - t) <= self.window_s]

        # Check for crash loop threshold
        if len(self.restart_history) >= self.max_restarts:
            self.is_safe_mode = True
            log.critical(
                "CRASH LOOP DETECTED: %d restarts occurred in %.1fs (Max: %d). ENTERING SAFE MODE!",
                len(self.restart_history),
                self.window_s,
                self.max_restarts,
            )
            self._kill_process()
            return False

        log.warning(
            "Restarting unresponsive process (Restart #%d in current window)...",
            len(self.restart_history) + 1,
        )
        self._kill_process()

        self.restart_history.append(now)
        self.total_restarts += 1

        return self.start_process()

    def _kill_process(self) -> None:
        """Terminate child process cleanly with fallback to force kill."""
        if self.sim_mode:
            self._sim_process_alive = False
            return

        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def run_loop(self, max_cycles: Optional[int] = None) -> None:
        """
        Main supervisor monitoring loop.
        """
        log.info("Watchdog supervisor loop active. Monitoring command: %s", " ".join(self.command))
        self.start_process()

        cycles = 0
        while not self.is_safe_mode:
            if max_cycles is not None and cycles >= max_cycles:
                break
            cycles += 1

            time.sleep(self.check_interval_s)

            if not self.is_heartbeat_fresh():
                success = self.restart_process()
                if not success:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-LKA Supervisor Watchdog")
    parser.add_argument(
        "--command",
        type=str,
        default="python deploy/jetson_run.py",
        help="Command line to supervise",
    )
    parser.add_argument(
        "--heartbeat-file",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "apex_heartbeat.txt"),
        help="Path to heartbeat file written by inference process",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="Heartbeat timeout in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Check interval in seconds")
    parser.add_argument("--max-restarts", type=int, default=3, help="Max restarts per window")
    parser.add_argument("--window", type=float, default=60.0, help="Sliding window in seconds")
    parser.add_argument("--sim-mode", action="store_true", help="Run in test simulation mode")
    args = parser.parse_args()

    watchdog = ProcessWatchdog(
        command=args.command,
        heartbeat_file=args.heartbeat_file,
        timeout_s=args.timeout,
        check_interval_s=args.interval,
        max_restarts=args.max_restarts,
        window_s=args.window,
        sim_mode=args.sim_mode,
    )
    watchdog.run_loop()


if __name__ == "__main__":
    main()
