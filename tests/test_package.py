import importlib.metadata
import subprocess
import sys

from kernelagent import __version__


def test_installed_package_and_cli():
    assert importlib.metadata.version("kernelagent-evidence") == __version__
    run = subprocess.run(
        [sys.executable, "-m", "kernelagent", "--version"], capture_output=True, text=True
    )
    assert run.returncode == 0
    assert run.stdout.strip() == __version__


def test_no_gpu_or_model_runtime_dependencies():
    dependencies = importlib.metadata.requires("kernelagent-evidence") or []
    assert dependencies == []
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kernelagent.cli, sys; "
            "assert not {'torch', 'triton', 'openai', 'anthropic'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
