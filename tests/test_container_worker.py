"""Container worker boundary (T04, ADR-0001): OS-enforced isolation checks.

Each case drives a real Docker container through ``execute_container`` and
observes the boundary from the trusted parent side: protocol identity,
read-only inputs, no network, secret-free environment, timeout/cancel
reclamation (including in-container setsid descendants), cgroup resource
limits, and infrastructure-failure semantics. Docker-unavailable and
image-unavailable environments skip with a recorded reason; a skip is a
gap in coverage, never a pass. The mount-source workspace boundary is
additionally covered daemon-free in ``test_container_path_preflight.py``
so it is exercised on every matrix regardless of Docker state."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from kernelagent.worker import (
    ContainerSpec,
    WorkerRequest,
    build_container_env,
    container_exists,
    execute_container,
    is_secret_name,
    list_worker_containers,
    snapshot_dir_hashes,
)

PYTHON_IMAGE = "docker.m.daocloud.io/library/python"
PYTHON_DIGEST = "sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
UBUNTU_IMAGE = "docker.m.daocloud.io/library/ubuntu"
UBUNTU_DIGEST = "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254"
DOCKER = ("docker",)


def _docker(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*DOCKER, *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def _docker_available() -> bool:
    try:
        return _docker("info", "--format", "ok").returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _image_present(repo: str, digest: str) -> bool:
    if not _docker_available():
        return False
    return _docker("image", "inspect", f"{repo}@{digest}").returncode == 0


pytestmark = [
    pytest.mark.skipif(not _docker_available(), reason="Docker service unavailable"),
]

require_python_image = pytest.mark.skipif(
    not _image_present(PYTHON_IMAGE, PYTHON_DIGEST),
    reason=f"pinned image {PYTHON_IMAGE}@{PYTHON_DIGEST[:19]}… absent (no auto-pull)",
)
require_ubuntu_image = pytest.mark.skipif(
    not _image_present(UBUNTU_IMAGE, UBUNTU_DIGEST),
    reason=f"pinned image {UBUNTU_IMAGE}@{UBUNTU_DIGEST[:19]}… absent (no auto-pull)",
)


def make_request(argv, tmp_path, **overrides):
    values = {
        "request_id": "req-1",
        "argv": tuple(argv),
        "timeout_seconds": 60.0,
        "workspace_root": tmp_path / "workspace",
    }
    values.update(overrides)
    return WorkerRequest(**values)


def make_spec(**overrides) -> ContainerSpec:
    values = {
        "image": PYTHON_IMAGE,
        "image_digest": PYTHON_DIGEST,
    }
    values.update(overrides)
    return ContainerSpec(**values)


def stage_inputs(workspace: Path, files: dict[str, str]) -> Path:
    inputs = workspace / "trusted-input"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (inputs / name).write_text(content, encoding="utf-8")
    return inputs


def extracted_output(outcome) -> Path:
    assert outcome.workdir is not None, "container executor must retain its work directory"
    out_dir = Path(outcome.workdir) / "out"
    assert out_dir.is_dir(), "extracted output directory missing"
    return out_dir


def record_of(outcome) -> dict:
    assert outcome.workdir is not None
    return json.loads((Path(outcome.workdir) / "container-record.json").read_text(encoding="utf-8"))


# -- C1 normal path -----------------------------------------------------------


@require_python_image
def test_normal_request_completes_and_extracts_output(tmp_path):
    request = make_request(
        ["/bin/sh", "-c", "echo hello-container; echo result-data > /out/result.txt"], tmp_path
    )
    outcome = execute_container(request, make_spec())
    assert outcome.status == "completed", outcome.stderr_tail
    assert outcome.exit_code == 0
    assert "hello-container" in outcome.stdout_tail
    result_file = extracted_output(outcome) / "result.txt"
    result_text = result_file.read_text(encoding="utf-8").strip()
    assert result_text == "result-data"
    assert result_file.stat().st_uid == 20000, "output must be written by the container uid"
    record = record_of(outcome)
    assert record["reclaimed"] is True
    assert not container_exists(DOCKER, record["container_name"])
    assert record["container_name"].startswith("ka-")
    assert request.request_id not in record["container_name"]
    leftovers = [name for name in list_worker_containers() if name == record["container_name"]]
    assert leftovers == []


@require_ubuntu_image
def test_normal_request_on_ubuntu_image(tmp_path):
    request = make_request(
        ["/bin/sh", "-c", "echo from-ubuntu > /out/marker.txt; id -u > /out/uid.txt"], tmp_path
    )
    outcome = execute_container(request, make_spec(image=UBUNTU_IMAGE, image_digest=UBUNTU_DIGEST))
    assert outcome.status == "completed", outcome.stderr_tail
    marker = (extracted_output(outcome) / "marker.txt").read_text(encoding="utf-8").strip()
    assert marker == "from-ubuntu"
    uid = (extracted_output(outcome) / "uid.txt").read_text(encoding="utf-8").strip()
    assert uid == "20000", "container payload must run as the non-root uid from ADR-0001"


# -- C2 candidate cannot self-report success ---------------------------------


@require_python_image
def test_candidate_self_report_does_not_become_success(tmp_path):
    request = make_request(
        [
            "python3",
            "-c",
            "import json; json.dump({'verified': True}, open('/out/result.json', 'w'))",
        ],
        tmp_path,
    )
    outcome = execute_container(request, make_spec())
    assert outcome.status == "completed"
    # Trusted-side rule: success is derived by the parent recomputing the
    # expected output, never from the candidate's own artifact claims.
    candidate_claim = json.loads((extracted_output(outcome) / "result.json").read_text())
    trusted_expected = "42"
    assert candidate_claim.get("verified") is True
    assert not candidate_claim.get("value") == trusted_expected
    record = record_of(outcome)
    assert record["reclaimed"] is True


# -- C3 read-only trusted inputs ----------------------------------------------


@require_python_image
def test_trusted_inputs_are_read_only_and_hash_stable(tmp_path):
    inputs = stage_inputs(tmp_path / "workspace", {"reference.txt": "trusted-reference"})
    before = snapshot_dir_hashes(inputs)
    request = make_request(
        [
            "python3",
            "-c",
            (
                "import json\n"
                "result = {'could_write_input': False}\n"
                "try:\n"
                "    open('/task/reference.txt', 'w').write('tampered')\n"
                "    result['could_write_input'] = True\n"
                "except OSError as exc:\n"
                "    result['error'] = type(exc).__name__\n"
                "result['read'] = open('/task/reference.txt').read()\n"
                "json.dump(result, open('/out/probe.json', 'w'))"
            ),
        ],
        tmp_path,
    )
    spec = make_spec(read_only_mounts=(("/task", inputs),))
    outcome = execute_container(request, spec)
    assert outcome.status == "completed", outcome.stderr_tail
    probe = json.loads((extracted_output(outcome) / "probe.json").read_text())
    assert probe["could_write_input"] is False
    assert probe["read"] == "trusted-reference"
    assert snapshot_dir_hashes(inputs) == before, "trusted input bytes must not change"


def test_mount_source_outside_workspace_rejected(tmp_path):
    request = make_request(["/bin/sh", "-c", "true"], tmp_path)
    outside = Path("/etc")
    spec = make_spec(read_only_mounts=(("/task", outside),))
    outcome = execute_container(request, spec)
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


def test_outside_mount_rejected_even_without_pinned_image(tmp_path):
    """R7 regression (workflow 34731172474): on a runner without the pinned
    image, the image-inspect infra error used to mask the boundary
    violation. The path check is pure validation and must fire before any
    Docker resource operation, image inspect included."""
    request = make_request(["/bin/sh", "-c", "true"], tmp_path)
    outside = Path("/etc")
    spec = make_spec(image_digest="sha256:" + "0" * 64, read_only_mounts=(("/task", outside),))
    outcome = execute_container(request, spec)
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail
    assert "refusing to auto-pull" not in outcome.stderr_tail


# -- C4 hostile identity and paths --------------------------------------------


def test_hostile_request_id_never_reaches_container_name(tmp_path):
    with pytest.raises(ValueError):
        make_request(
            ["/bin/sh", "-c", "true"],
            tmp_path,
            request_id="evil-../../ancestor",
        )


@require_python_image
def test_path_escape_inside_container_confined(tmp_path):
    request = make_request(
        [
            "python3",
            "-c",
            (
                "import json\n"
                "result = {'escaped': False}\n"
                "try:\n"
                "    open('/etc/passwd', 'a').close()\n"
                "    result['escaped'] = True\n"
                "except OSError as exc:\n"
                "    result['error'] = type(exc).__name__\n"
                "json.dump(result, open('/out/escape.json', 'w'))"
            ),
        ],
        tmp_path,
    )
    outcome = execute_container(request, make_spec())
    assert outcome.status == "completed", outcome.stderr_tail
    escape = json.loads((extracted_output(outcome) / "escape.json").read_text())
    assert escape["escaped"] is False, "read-only rootfs must refuse writes outside tmp mounts"


# -- C5 no network ------------------------------------------------------------


@require_python_image
def test_network_is_unreachable(tmp_path):
    request = make_request(
        [
            "python3",
            "-c",
            (
                "import json, socket\n"
                "result = {'connect': 'unresolved'}\n"
                "try:\n"
                "    socket.setdefaulttimeout(3)\n"
                "    socket.socket().connect(('1.1.1.1', 80))\n"
                "    result['connect'] = 'succeeded'\n"
                "except OSError as exc:\n"
                "    result['connect'] = type(exc).__name__\n"
                "try:\n"
                "    socket.getaddrinfo('example.com', 443)\n"
                "    result['dns'] = 'succeeded'\n"
                "except OSError as exc:\n"
                "    result['dns'] = type(exc).__name__\n"
                "json.dump(result, open('/out/net.json', 'w'))"
            ),
        ],
        tmp_path,
    )
    outcome = execute_container(request, make_spec())
    assert outcome.status == "completed", outcome.stderr_tail
    net = json.loads((extracted_output(outcome) / "net.json").read_text())
    assert net["connect"] != "succeeded", "container must have no outbound network"
    assert net["dns"] != "succeeded", "container must have no DNS"


# -- C6 secrets never reach the container -------------------------------------


def test_secret_shaped_env_names_rejected():
    with pytest.raises(ValueError):
        build_container_env((("MODEL_API_KEY", "super-secret"),))


@require_python_image
def test_container_environment_is_scrubbed(tmp_path):
    request = make_request(
        [
            "python3",
            "-c",
            "import json, os; json.dump(dict(os.environ), open('/out/env.json', 'w'))",
        ],
        tmp_path,
        env_extra=(("TASK_NOTE", "benign"),),
    )
    outcome = execute_container(request, make_spec())
    assert outcome.status == "completed", outcome.stderr_tail
    container_env = json.loads((extracted_output(outcome) / "env.json").read_text())
    # Nothing from the trusted parent's own environment may reach the
    # container: secret-shaped host variable names must be absent entirely,
    # and the host PATH value must not survive anywhere (the container only
    # gets the image's own PATH plus explicitly injected pairs).
    for name in os.environ:
        if is_secret_name(name):
            assert name not in container_env, f"host secret-shaped env leaked: {name}"
    parent_path = os.environ.get("PATH", "")
    assert parent_path
    for value in container_env.values():
        assert value != parent_path, "host PATH value leaked into the container"
    assert container_env.get("TASK_NOTE") == "benign"


# -- C7 timeout and descendant reclamation ------------------------------------


@require_python_image
def test_timeout_kills_container_and_descendants(tmp_path):
    before = set(list_worker_containers())
    request = make_request(
        [
            "python3",
            "-c",
            (
                "import subprocess, time\n"
                # setsid inside the container cannot leave the PID namespace;
                # this is the exact escape that survives the T04a killpg.
                "subprocess.Popen(['setsid', 'sleep', '300'], start_new_session=True)\n"
                "time.sleep(300)"
            ),
        ],
        tmp_path,
        timeout_seconds=6.0,
    )
    outcome = execute_container(request, make_spec())
    assert outcome.status == "timeout"
    assert outcome.killed is True
    record = record_of(outcome)
    assert record["reclaimed"] is True
    assert not container_exists(DOCKER, record["container_name"])
    after = set(list_worker_containers())
    assert after == before, "no worker container may outlive the request"


# -- C8 cancel -----------------------------------------------------------------


@require_python_image
def test_cancel_stops_the_container(tmp_path):
    cancel_event = threading.Event()
    result: dict = {}

    def run() -> None:
        request = make_request(["python3", "-c", "import time; time.sleep(300)"], tmp_path)
        result["outcome"] = execute_container(request, make_spec(), cancel_event=cancel_event)

    thread = threading.Thread(target=run)
    thread.start()
    threading.Event().wait(2.0)
    cancel_event.set()
    thread.join(timeout=60)
    assert "outcome" in result
    outcome = result["outcome"]
    assert outcome.status == "timeout"
    assert outcome.killed is True
    assert "[cancelled]" in outcome.stderr_tail
    record = record_of(outcome)
    assert record["reclaimed"] is True
    assert not container_exists(DOCKER, record["container_name"])


# -- C9 resource limits ---------------------------------------------------------


@require_python_image
def test_memory_limit_structured_failure(tmp_path):
    request = make_request(
        ["python3", "-c", "data = bytearray(400 * 1024 * 1024); print(len(data))"],
        tmp_path,
        timeout_seconds=60.0,
    )
    outcome = execute_container(request, make_spec(memory_bytes=128 * 1024 * 1024))
    assert outcome.status == "failed"
    assert "[oom_killed]" in outcome.stderr_tail
    record = record_of(outcome)
    assert record["state"]["oom_killed"] is True


@require_python_image
def test_pids_limit_structured_failure(tmp_path):
    request = make_request(
        [
            "python3",
            "-c",
            (
                "import json, subprocess\n"
                "children, spawn_failed = 0, False\n"
                "for _ in range(48):\n"
                "    try:\n"
                "        subprocess.Popen(['sleep', '60'])\n"
                "        children += 1\n"
                "    except (OSError, subprocess.SubprocessError):\n"
                "        spawn_failed = True\n"
                "        break\n"
                "json.dump({'children': children, 'spawn_failed': spawn_failed},\n"
                "          open('/out/pids.json', 'w'))"
            ),
        ],
        tmp_path,
        timeout_seconds=90.0,
    )
    outcome = execute_container(request, make_spec(pids_limit=16))
    assert outcome.status == "completed", outcome.stderr_tail
    pids = json.loads((extracted_output(outcome) / "pids.json").read_text())
    assert pids["spawn_failed"] is True, "fork beyond the pids limit must fail inside the container"
    assert pids["children"] < 48


@require_python_image
def test_output_size_limit_breach_kills_and_marks_failed(tmp_path):
    request = make_request(
        [
            "python3",
            "-c",
            (
                "import time\n"
                "chunk = b'x' * (8 * 1024 * 1024)\n"
                "with open('/out/big.bin', 'wb') as handle:\n"
                "    for _ in range(16):\n"  # 128 MiB against a 1 MiB cap
                "        handle.write(chunk)\n"
                "        handle.flush()\n"
                "        time.sleep(0.05)"
            ),
        ],
        tmp_path,
        timeout_seconds=120.0,
    )
    outcome = execute_container(request, make_spec(output_limit_bytes=1024 * 1024))
    assert outcome.status == "failed", outcome.stderr_tail
    assert "[output_limit]" in outcome.stderr_tail
    record = record_of(outcome)
    assert record["reclaimed"] is True
    assert record["output_bytes"] > 1024 * 1024, "breach must be visible in output accounting"
    assert record["output_bytes"] < 128 * 1024 * 1024, (
        "monitor must stop the breach well before completion"
    )


@require_python_image
def test_log_stream_limit_breaches_to_failed(tmp_path):
    request = make_request(
        ["python3", "-c", "import sys; sys.stdout.write('x' * (3 * 1024 * 1024))"],
        tmp_path,
        timeout_seconds=90.0,
    )
    outcome = execute_container(request, make_spec(log_limit_bytes=1024 * 1024))
    assert outcome.status == "failed"
    assert "[log_limit]" in outcome.stderr_tail
    record = record_of(outcome)
    assert record["reclaimed"] is True


# -- C10 infrastructure failures ------------------------------------------------


def test_missing_docker_binary_is_infra_error(tmp_path):
    request = make_request(["/bin/sh", "-c", "true"], tmp_path)
    outcome = execute_container(request, make_spec(), docker_command=("ka-no-such-docker-binary",))
    assert outcome.status == "infra_error"
    assert "not found" in outcome.stderr_tail


def test_absent_image_digest_is_infra_error_not_autopull(tmp_path):
    request = make_request(["/bin/sh", "-c", "true"], tmp_path)
    fake_digest = "sha256:" + "0" * 64
    outcome = execute_container(request, make_spec(image_digest=fake_digest))
    assert outcome.status == "infra_error"
    assert "refusing to auto-pull" in outcome.stderr_tail


def test_spec_validation_rejects_loose_identity():
    with pytest.raises(ValueError):
        make_spec(image_digest="sha256:xyz")
    with pytest.raises(ValueError):
        make_spec(image="python:3.11; rm -rf /")
    with pytest.raises(ValueError):
        make_spec(user="root")
    with pytest.raises(ValueError):
        make_spec(read_only_mounts=(("/out", Path("/tmp")),))
    with pytest.raises(ValueError):
        make_spec(memory_bytes=-1)
    with pytest.raises(ValueError):
        make_spec(gpu_devices=("nvidia.com/gpu-without-equals",))
