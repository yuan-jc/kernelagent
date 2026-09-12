"""Worker boundary: execution, timeout tree-kill, env scrubbing, dirs."""

import json
import sys
import time

import pytest

from kernelagent.worker import (
    PathOutsideWorkspaceError,
    TrustedDirModifiedError,
    WorkerOutcome,
    WorkerRequest,
    assert_dir_unchanged,
    build_child_env,
    ensure_within,
    execute,
    process_exists,
    snapshot_dir_hashes,
)

PYTHON = sys.executable


def make_request(argv, tmp_path, **overrides):
    values = {
        "request_id": "req-1",
        "argv": tuple(argv),
        "timeout_seconds": 30.0,
        "workspace_root": tmp_path / "workspace",
    }
    values.update(overrides)
    return WorkerRequest(**values)


# -- request/outcome contracts -----------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": ""},
        {"argv": ()},
        {"argv": ("python", "")},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1.0},
        {"timeout_seconds": float("nan")},
        {"workspace_root": "not-a-path"},
        {"env_extra": (("A=BAD", "v"),)},
        {"env_extra": (("OK", 3),)},
        {"output_tail_chars": 0},
    ],
    ids=[
        "id",
        "empty-argv",
        "empty-arg",
        "zero-timeout",
        "neg-timeout",
        "nan-timeout",
        "str-root",
        "eq-in-name",
        "non-str-value",
        "zero-tail",
    ],
)
def test_invalid_requests_rejected(tmp_path, overrides):
    argv = overrides.pop("argv", (PYTHON, "-c", "pass"))
    with pytest.raises(ValueError):
        make_request(argv, tmp_path, **overrides)


def test_outcome_status_and_types_validated():
    with pytest.raises(ValueError):
        WorkerOutcome("r", "excellent", 0, 1.0, "", "", False)
    with pytest.raises(ValueError):
        WorkerOutcome("r", "completed", True, 1.0, "", "", False)


# -- normal path -------------------------------------------------------------


def test_normal_payload_completes_and_cleans_workdir(tmp_path):
    (tmp_path / "workspace").mkdir()
    outcome = execute(make_request((PYTHON, "-c", "print('hello worker')"), tmp_path))
    assert outcome.status == "completed"
    assert outcome.exit_code == 0
    assert outcome.killed is False
    assert "hello worker" in outcome.stdout_tail
    assert outcome.duration_seconds > 0
    assert outcome.workdir is None
    assert not any((tmp_path / "workspace").iterdir())


# -- timeout and process tree ------------------------------------------------


def _spawn_tree_script():
    # The pid-file path arrives as a real argument (sys.argv[1]); nothing is
    # embedded into the source text.
    return (
        "import os, subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', "
        "\"import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); "
        'time.sleep(120)", sys.argv[1]])\n'
        "open(sys.argv[1] + '.child', 'w').write(str(os.getpid()))\n"
        "gc.wait()\n"
    )


def _wait_gone(pid: int, grace_seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.2)
    return False


def test_timeout_kills_whole_process_tree(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid_file = workspace / "pids.txt"
    argv = (PYTHON, "-c", _spawn_tree_script(), str(pid_file))
    outcome = execute(make_request(argv, tmp_path, timeout_seconds=4.0))
    assert outcome.status == "timeout"
    assert outcome.killed is True
    assert outcome.exit_code is not None  # killed child reports a code on reaping
    assert pid_file.is_file()
    child_pid_file = workspace / (pid_file.name + ".child")
    assert child_pid_file.is_file()
    pids = [
        int(pid_file.read_text(encoding="utf-8").strip()),  # grandchild
        int(child_pid_file.read_text(encoding="utf-8").strip()),  # child
    ]
    for pid in pids:
        assert _wait_gone(pid), f"pid {pid} survived the tree kill"


def test_failed_payload_reports_stderr_and_exit_code(tmp_path):
    outcome = execute(
        make_request(
            (PYTHON, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"), tmp_path
        )
    )
    assert outcome.status == "failed"
    assert outcome.exit_code == 3
    assert "boom" in outcome.stderr_tail


def test_forged_success_output_still_failed_by_exit_code(tmp_path):
    outcome = execute(
        make_request((PYTHON, "-c", "print('SUCCESS'); import sys; sys.exit(3)"), tmp_path)
    )
    assert outcome.status == "failed"
    assert "SUCCESS" in outcome.stdout_tail


def test_keep_workdir_on_failure_policy(tmp_path):
    outcome = execute(
        make_request(
            (PYTHON, "-c", "open('leftover.txt', 'w').write('x'); import sys; sys.exit(1)"),
            tmp_path,
            keep_workdir_on_failure=True,
        )
    )
    assert outcome.status == "failed"
    assert outcome.workdir is not None
    assert (
        outcome.workdir and (__import__("pathlib").Path(outcome.workdir) / "leftover.txt").is_file()
    )


# -- environment scrubbing ---------------------------------------------------


def test_build_child_env_denies_secret_shaped_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    monkeypatch.setenv("MY_SECRET_TOKEN", "leak")
    monkeypatch.setenv("DB_PASSWORD", "leak")
    monkeypatch.setenv("KA_PLAIN_VAR", "fine")
    env = build_child_env(extra=(("KA_EXTRA", "allowed"),))
    assert "OPENAI_API_KEY" not in env
    assert "MY_SECRET_TOKEN" not in env
    assert "DB_PASSWORD" not in env
    assert env["KA_PLAIN_VAR"] == "fine"
    assert env["KA_EXTRA"] == "allowed"


def test_build_child_env_rejects_secret_shaped_extras():
    with pytest.raises(ValueError, match="denylist"):
        build_child_env(extra=(("KA_API_KEY", "nope"),))


def test_child_environment_is_scrubbed_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv("KA_SECRET_VALUE", "leak-me")
    script = (
        "import json, os; print(json.dumps({"
        "'marker': os.environ.get('KA_TEST_MARKER'), "
        "'secret': os.environ.get('KA_SECRET_VALUE')}))"
    )
    outcome = execute(
        make_request(
            (PYTHON, "-c", script),
            tmp_path,
            env_extra=(("KA_TEST_MARKER", "visible"),),
        )
    )
    payload = json.loads(outcome.stdout_tail)
    assert payload["marker"] == "visible"
    assert payload["secret"] is None


# -- directory boundary ------------------------------------------------------


def test_ensure_within_rejects_escapes(tmp_path):
    base = tmp_path / "workspace"
    base.mkdir()
    assert ensure_within(base, base / "sub" / "file.txt") == (base / "sub" / "file.txt").resolve()
    for escape in (
        base / ".." / "outside.txt",
        tmp_path / "sibling.txt",
        base / "sub" / ".." / ".." / "up.txt",
    ):
        with pytest.raises(PathOutsideWorkspaceError):
            ensure_within(base, escape)


def test_trusted_dir_tampering_is_detected(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "reference.json").write_text("{}", encoding="utf-8")
    before = snapshot_dir_hashes(trusted)
    payload = "import sys; open(sys.argv[1], 'a', encoding='utf-8').write('tampered'); sys.exit(0)"
    outcome = execute(
        make_request((PYTHON, "-c", payload, str(trusted / "reference.json")), tmp_path)
    )
    assert outcome.status == "completed"  # the payload itself succeeded
    after = snapshot_dir_hashes(trusted)
    with pytest.raises(TrustedDirModifiedError, match="tampered|reference"):
        assert_dir_unchanged(before, after)


def test_benign_run_leaves_trusted_dir_unchanged(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "reference.json").write_text("{}", encoding="utf-8")
    before = snapshot_dir_hashes(trusted)
    execute(make_request((PYTHON, "-c", "print('benign')"), tmp_path))
    assert_dir_unchanged(before, snapshot_dir_hashes(trusted))


def test_snapshot_hash_changes_with_content(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("one", encoding="utf-8")
    first = snapshot_dir_hashes(tmp_path)
    target.write_text("two", encoding="utf-8")
    second = snapshot_dir_hashes(tmp_path)
    assert first != second
    assert assert_dir_unchanged(first, first) is None


# -- import boundary ---------------------------------------------------------


def test_worker_imports_no_gpu_or_third_party_helpers():
    code = (
        "import sys; import kernelagent.worker; "
        "bad = {'torch', 'triton', 'psutil', 'openai', 'anthropic'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = __import__("subprocess").run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


# -- review repros (artifacts/review-t04a/review.md): each must fail on the
#    unfixed implementation and pass after the fix -------------------------


def test_repro_request_id_traversal_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_request((PYTHON, "-c", "pass"), tmp_path, request_id="x/../../escaped")
    with pytest.raises(ValueError):
        make_request((PYTHON, "-c", "pass"), tmp_path, request_id="a\b")


def test_repro_workdir_stays_inside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outcome = execute(
        make_request((PYTHON, "-c", "print('ok')"), tmp_path, keep_workdir_on_failure=False)
    )
    assert outcome.status == "completed"
    for entry in workspace.iterdir():
        assert "payload-" in entry.name


def test_repro_descendants_die_when_payload_exits_early(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    late_file = workspace / "late.txt"
    payload = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"\"import time, sys; time.sleep(2); open(sys.argv[1], 'w').write('late')\", "
        f"{str(late_file)!r}])\n"
    )
    outcome = execute(make_request((PYTHON, "-c", payload), tmp_path, timeout_seconds=10))
    assert outcome.status == "completed"
    time.sleep(3.5)  # past the descendant's 2s write delay
    assert not late_file.exists(), "descendant outlived the request budget"


def test_repro_log_tail_does_not_load_whole_file(tmp_path, monkeypatch):
    marker = "END-OF-LOG"
    payload = "import sys; sys.stdout.write('x' * 4194304 + %r)" % marker

    def _boom(self, *args, **kwargs):
        raise AssertionError("unbounded read_bytes used for tail read")

    monkeypatch.setattr("pathlib.Path.read_bytes", _boom)
    outcome = execute(make_request((PYTHON, "-c", payload), tmp_path, output_tail_chars=16))
    assert outcome.status == "completed"
    assert outcome.stdout_tail.endswith(marker)


def test_repro_missing_executable_returns_infra_error(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outcome = execute(make_request(("definitely-missing-binary-xyz",), tmp_path))
    assert outcome.status == "infra_error"
    assert outcome.workdir is None
    assert not any(workspace.iterdir()), "infra failure must clean its payload directory"


def test_repro_cleanup_failure_is_reported(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: None)
    outcome = execute(
        make_request(
            (PYTHON, "-c", "open('leftover.txt', 'w').write('x'); import sys; sys.exit(1)"),
            tmp_path,
        )
    )
    assert outcome.status == "failed"
    assert outcome.workdir is not None, "undeleted workdir must be reported, not claimed clean"


def test_repro_env_extra_outer_container_must_be_tuple(tmp_path):
    with pytest.raises(ValueError):
        make_request((PYTHON, "-c", "pass"), tmp_path, env_extra=[("A", "1")])


def test_repro_keep_workdir_must_be_bool(tmp_path):
    with pytest.raises(ValueError):
        make_request((PYTHON, "-c", "pass"), tmp_path, keep_workdir_on_failure="yes")


def test_repro_nul_bytes_rejected_in_request(tmp_path):
    with pytest.raises(ValueError):
        make_request((PYTHON + "\0bad", "-c", "pass"), tmp_path)
    with pytest.raises(ValueError):
        make_request((PYTHON, "-c", "pass"), tmp_path, env_extra=(("A", "v\0"),))
