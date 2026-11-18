import subprocess

import scheduler as sch


class Result:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_run_script_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result(0))
    assert sch._run_script("parser.py", timeout=10) is True


def test_run_script_failure_logs(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result(2, "boom"))
    assert sch._run_script("parser.py", timeout=10) is False


def test_run_script_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="parser.py", timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert sch._run_script("parser.py", timeout=1) is False


def test_run_script_unexpected_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "run", raise_error)
    assert sch._run_script("parser.py", timeout=1) is False


def test_pipeline_stops_when_parser_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(sch, "_run_script", lambda name, timeout: calls.append(name) or False)
    monkeypatch.setattr(sch, "_reload_callback", lambda: calls.append("reload"))

    sch._run_pipeline_subprocess()

    assert calls == ["parser.py"]


def test_pipeline_runs_index_and_reload(monkeypatch):
    calls = []
    monkeypatch.setattr(sch, "_run_script", lambda name, timeout: calls.append(name) or True)
    monkeypatch.setattr(sch, "_reload_callback", lambda: calls.append("reload"))

    sch._run_pipeline_subprocess()

    assert calls == ["parser.py", "create_index.py", "reload"]


def test_pipeline_skips_when_already_running(monkeypatch):
    calls = []
    monkeypatch.setattr(sch, "_run_script", lambda name, timeout: calls.append(name) or True)
    sch._pipeline_lock.acquire()
    try:
        sch._run_pipeline_subprocess()
    finally:
        sch._pipeline_lock.release()
    assert calls == []


def test_pipeline_survives_reload_error(monkeypatch):
    monkeypatch.setattr(sch, "_run_script", lambda name, timeout: True)

    def broken_reload():
        raise RuntimeError("reload failed")

    monkeypatch.setattr(sch, "_reload_callback", broken_reload)
    sch._run_pipeline_subprocess()
