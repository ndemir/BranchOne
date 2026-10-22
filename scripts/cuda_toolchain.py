"""Select CUDA 12.8 for extension builds, independently of the runtime driver."""

import os
import re
import subprocess
from pathlib import Path

PACKAGES = [
    "cuda-nvcc-12-8=12.8.93-1",
    "cuda-cudart-dev-12-8=12.8.90-1",
    "cuda-cccl-12-8=12.8.90-1",
]


def compiler_version(home):
    compiler = Path(home) / "bin/nvcc"
    if not compiler.is_file():
        return None
    result = subprocess.run(
        [str(compiler), "--version"], capture_output=True, text=True, check=True
    )
    match = re.search(r"release (\d+\.\d+),", result.stdout)
    return match.group(1) if match else None


def configure(*, install=False):
    candidates = [os.environ.get("CUDA_HOME"), "/usr/local/cuda-12.8"]
    selected = next(
        (Path(p) for p in candidates if p and compiler_version(p) == "12.8"),
        None,
    )
    if selected is None and install:
        if os.geteuid() != 0:
            raise RuntimeError(
                "Install CUDA 12.8 development tools and set CUDA_HOME before setup"
            )
        # Colab provides NVIDIA's signed Ubuntu repository. Install compiler and
        # development headers only; leave the runtime's GPU driver in place.
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(
            ["apt-get", "install", "-y", "--no-install-recommends", *PACKAGES],
            check=True,
            env=os.environ | {"DEBIAN_FRONTEND": "noninteractive"},
        )
        selected = Path("/usr/local/cuda-12.8")
    if selected is None or compiler_version(selected) != "12.8":
        raise RuntimeError(
            "The extension requires a CUDA 12.8 compiler to match Torch cu128; "
            "run setup_colab.py or set CUDA_HOME to a CUDA 12.8 toolkit"
        )
    os.environ["CUDA_HOME"] = str(selected)
    os.environ["PATH"] = str(selected / "bin") + os.pathsep + os.environ["PATH"]
    version = subprocess.check_output(
        [str(selected / "bin/nvcc"), "--version"], text=True
    )
    return {"cuda_home": str(selected), "nvcc": version}
