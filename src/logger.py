"""
logger.py — JSON logging for all output files.

All output files are written incrementally so data is never lost
if the simulation crashes mid-run.
"""
import json
import os
from pathlib import Path
from typing import Any


class SimLogger:
    """Manages all four output JSON files."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._orders_path    = self.output_dir / "orders_log.json"
        self._snapshots_path = self.output_dir / "portfolio_snapshots.json"
        self._llm_path       = self.output_dir / "llm_call_log.json"
        self._results_path   = self.output_dir / "results.json"

        # In-memory lists — flushed to disk after each update
        self._orders:    list[dict] = []
        self._snapshots: list[dict] = []
        self._llm_calls: list[dict] = []

        # Initialise files as empty arrays
        self._flush(self._orders_path,    self._orders)
        self._flush(self._snapshots_path, self._snapshots)
        self._flush(self._llm_path,       self._llm_calls)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flush(path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_fill(self, fill: dict) -> None:
        """Append a fill record to orders_log.json."""
        self._orders.append(fill)
        self._flush(self._orders_path, self._orders)

    def log_snapshot(self, snapshot: dict) -> None:
        """Append a portfolio snapshot to portfolio_snapshots.json."""
        self._snapshots.append(snapshot)
        self._flush(self._snapshots_path, self._snapshots)

    def log_llm_call(self, record: dict) -> None:
        """Append an LLM call record to llm_call_log.json."""
        self._llm_calls.append(record)
        self._flush(self._llm_path, self._llm_calls)

    def write_results(self, results: dict) -> None:
        """Write final results.json (overwrites)."""
        self._flush(self._results_path, results)
        from src.log import vprint
        vprint(f"[Logger] Results written to {self._results_path}")

    # ------------------------------------------------------------------
    # Accessors (for metrics)
    # ------------------------------------------------------------------
    @property
    def orders(self) -> list[dict]:
        return self._orders

    @property
    def snapshots(self) -> list[dict]:
        return self._snapshots

    @property
    def llm_calls(self) -> list[dict]:
        return self._llm_calls
