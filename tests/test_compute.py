from zenreg import available_cpu_count, print_available_compute


def test_available_cpu_count_returns_positive_integer():
    assert isinstance(available_cpu_count(), int)
    assert available_cpu_count() >= 1


def test_print_available_compute_reports_n_jobs_hint(capsys):
    n_cpus = print_available_compute()
    captured = capsys.readouterr()

    assert n_cpus == available_cpu_count()
    assert "ZenReg available CPU workers" in captured.out
    assert "n_jobs=-1" in captured.out
