"""
Optional memory profiling helpers for ZenReg workflows.

Author: Fabrizio Musacchio
Date: July 2026
"""

from __future__ import annotations

import csv
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _import_psutil():
    """Import psutil lazily because memory profiling is an optional diagnostic."""

    try:
        import psutil
    except ImportError as exc:
        raise ImportError(
            "ZenReg memory profiling requires psutil. Install psutil or use an "
            "environment such as zenreg2 that already provides it."
        ) from exc
    return psutil


@dataclass
class MemoryTracker:
    """
    Sample process memory over time and annotate ZenReg processing steps.

    The tracker records resident set size (RSS) from the operating-system
    process. This captures NumPy/SciPy/SimpleITK/native allocations better than
    Python-only tracemalloc and is therefore the more relevant metric for large
    image stacks.
    """

    csv_path: str | Path | None = None
    plot_path: str | Path | None = None
    interval_s: float = 0.10
    enabled: bool = True
    label: str = "zenreg"
    _records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _start_time: float | None = field(default=None, init=False)
    _process: Any = field(default=None, init=False)

    def start(self):
        """Start background RSS sampling."""

        if not self.enabled:
            return self
        if self._thread is not None:
            return self
        psutil = _import_psutil()
        self._process = psutil.Process(os.getpid())
        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self.mark("tracker:start")
        self._thread = threading.Thread(target=self._sample_loop, name="ZenRegMemoryTracker", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop sampling and write configured CSV/plot outputs."""

        if not self.enabled:
            return self
        if self._thread is not None:
            self.mark("tracker:stop")
            self._stop_event.set()
            self._thread.join(timeout=max(1.0, float(self.interval_s) * 4.0))
            self._thread = None
        if self.csv_path is not None:
            self.write_csv(self.csv_path)
        if self.plot_path is not None:
            self.write_plot(self.plot_path)
        return self

    def mark(self, step: str):
        """Record an explicit step marker with a synchronous memory sample."""

        if not self.enabled:
            return
        self._record("mark", str(step))

    @contextmanager
    def step(self, step_name: str):
        """Context manager that records ``step:start`` and ``step:end`` markers."""

        self.mark(f"{step_name}:start")
        try:
            yield self
        finally:
            self.mark(f"{step_name}:end")

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a detached copy of all recorded samples and marks."""

        with self._lock:
            return [dict(record) for record in self._records]

    def summary(self) -> dict[str, float]:
        """Return peak and final memory values in MB."""

        records = self.records
        if not records:
            return {}
        rss_values = [float(record["rss_mb"]) for record in records]
        vms_values = [float(record["vms_mb"]) for record in records]
        return {
            "peak_rss_mb": max(rss_values),
            "final_rss_mb": rss_values[-1],
            "peak_vms_mb": max(vms_values),
            "final_vms_mb": vms_values[-1],
            "duration_s": float(records[-1]["elapsed_s"]),
        }

    def write_csv(self, path: str | Path):
        """Write the trace table to CSV."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.records
        fieldnames = [
            "elapsed_s",
            "event",
            "step",
            "rss_mb",
            "vms_mb",
            "uss_mb",
            "thread_count",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    def write_plot(self, path: str | Path):
        """Write a memory trace plot with vertical step markers."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zenreg-matplotlib"))

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        rows = self.records
        if not rows:
            raise RuntimeError("MemoryTracker has no records to plot.")
        times = [float(row["elapsed_s"]) for row in rows]
        rss = [float(row["rss_mb"]) for row in rows]
        uss = [row.get("uss_mb") for row in rows]
        has_uss = any(value not in (None, "") for value in uss)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times, rss, color="#2f5597", linewidth=1.8, label="RSS MB")
        if has_uss:
            ax.plot(
                times,
                [float(value) if value not in (None, "") else float("nan") for value in uss],
                color="#70ad47",
                linewidth=1.2,
                label="USS MB",
            )
        mark_rows = [row for row in rows if row.get("event") == "mark"]
        y_top = max(rss) if rss else 1.0
        for index, row in enumerate(mark_rows):
            x = float(row["elapsed_s"])
            ax.axvline(x, color="#a5a5a5", linewidth=0.6, alpha=0.55)
            if index % 2 == 0:
                ax.text(
                    x,
                    y_top,
                    str(row.get("step", "")),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                    color="#595959",
                )
        ax.set_title(f"ZenReg memory trace: {self.label}")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Memory (MB)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(float(self.interval_s)):
            self._record("sample", "sample")

    def _record(self, event: str, step: str) -> None:
        if self._process is None:
            return
        now = time.perf_counter()
        start = self._start_time if self._start_time is not None else now
        info = self._process.memory_info()
        uss_mb = ""
        try:
            full_info = self._process.memory_full_info()
            uss_mb = float(getattr(full_info, "uss")) / 1024.0 / 1024.0
        except Exception:
            uss_mb = ""
        record = {
            "elapsed_s": float(now - start),
            "event": str(event),
            "step": str(step),
            "rss_mb": float(info.rss) / 1024.0 / 1024.0,
            "vms_mb": float(info.vms) / 1024.0 / 1024.0,
            "uss_mb": uss_mb,
            "thread_count": int(self._process.num_threads()),
        }
        with self._lock:
            self._records.append(record)


@contextmanager
def profile_memory(
    *,
    csv_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    interval_s: float = 0.10,
    enabled: bool = True,
    label: str = "zenreg",
):
    """Convenience context manager for one profiled ZenReg workflow."""

    tracker = MemoryTracker(
        csv_path=csv_path,
        plot_path=plot_path,
        interval_s=interval_s,
        enabled=enabled,
        label=label,
    )
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


__all__ = ["MemoryTracker", "profile_memory"]
