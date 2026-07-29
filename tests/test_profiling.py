from pathlib import Path

from zenreg.profiling import MemoryTracker


def test_memory_tracker_writes_csv_and_plot(tmp_path):
    tracker = MemoryTracker(
        csv_path=tmp_path / "trace.csv",
        plot_path=tmp_path / "trace.png",
        interval_s=0.01,
        label="test",
    )
    tracker.start()
    tracker.mark("unit:test")
    data = [index**2 for index in range(1000)]
    assert data[-1] == 998001
    tracker.stop()

    csv_path = Path(tmp_path / "trace.csv")
    plot_path = Path(tmp_path / "trace.png")
    assert csv_path.exists()
    assert plot_path.exists()
    assert "unit:test" in csv_path.read_text(encoding="utf-8")
    assert plot_path.stat().st_size > 0


def test_memory_tracker_disabled_is_noop(tmp_path):
    tracker = MemoryTracker(
        csv_path=tmp_path / "trace.csv",
        plot_path=tmp_path / "trace.png",
        enabled=False,
    )
    tracker.start()
    tracker.mark("ignored")
    tracker.stop()

    assert tracker.records == []
    assert not (tmp_path / "trace.csv").exists()
    assert not (tmp_path / "trace.png").exists()
