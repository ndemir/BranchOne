"""Build the pinned fast convolution extension in the selected Colab environment."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from cuda_toolchain import configure

SOURCE_URL = "https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/causal_conv1d-1.7.0.tar.gz"
SOURCE_SHA256 = "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, help="Build cache outside the public repository"
    )
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    compiler = configure()
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    import torch

    if not torch.cuda.is_available() or torch.__version__ != "2.11.0+cu128":
        raise RuntimeError(
            "Use the recorded Torch CUDA 12.8 environment on the remote GPU"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "causal_conv1d-1.7.0.tar.gz"
    if not source.exists():
        urllib.request.urlretrieve(SOURCE_URL, source)
    if digest(source) != SOURCE_SHA256:
        raise RuntimeError("Convolution source checksum mismatch")
    env = os.environ | {
        "MAX_JOBS": "4",
        "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--verbose",
            "--no-build-isolation",
            "--no-deps",
            str(source),
            "--wheel-dir",
            str(output),
        ],
        check=True,
        env=env,
    )
    wheels = list(output.glob("causal_conv1d-1.7.0-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            "Expected one wheel; use a clean build directory per Python/CUDA environment"
        )
    if args.install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(wheels[0])],
            check=True,
        )
    record = {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "wheel_sha256": digest(wheels[0]),
        "wheel": wheels[0].name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "installed": args.install,
        "compiler": compiler,
    }
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
