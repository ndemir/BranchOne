"""Install only the explicitly pinned training dependencies into a fresh environment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cuda_toolchain import configure

TORCH_PACKAGES = [
    "torch==2.11.0+cu128",
    "torchvision==0.26.0+cu128",
    "torchaudio==2.11.0+cu128",
    "torchcodec==0.11.0+cu128",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-cache", required=True, help="External wheel/source cache"
    )
    parser.add_argument(
        "--environment-record",
        required=True,
        help="Write outside the public source tree",
    )
    parser.add_argument(
        "--venv", required=True, help="New isolated environment outside the repository"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = configure(install=True)

    def run(command):
        return subprocess.run(command, check=True)

    run([sys.executable, "-m", "pip", "install", "uv==0.12.17"])
    environment = os.environ | {
        "UV_PROJECT_ENVIRONMENT": str(Path(args.venv).resolve())
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uv",
            "sync",
            "--project",
            str(root),
            "--locked",
            "--extra",
            "build-kernels",
            "--extra",
            "test",
            "--python",
            sys.executable,
        ],
        env=environment,
        check=True,
    )
    python = str(Path(args.venv).resolve() / "bin/python")
    run(
        [
            python,
            str(root / "scripts/build_causal_conv.py"),
            "--output",
            args.kernel_cache,
            "--install",
        ]
    )
    run(
        [
            python,
            "-c",
            "import branchone.training, branchone.shared_training; print('TRAINING IMPORTS PASSED')",
        ]
    )
    names = [
        "torch",
        "torchvision",
        "torchaudio",
        "torchcodec",
        "transformers",
        "peft",
        "accelerate",
        "flash-linear-attention",
        "fla-core",
        "numpy",
        "Pillow",
        "safetensors",
        "causal-conv1d",
    ]
    code = (
        "import importlib.metadata as md,json,sys; print(json.dumps({'python':sys.version.split()[0], 'packages':{name:md.version(name) for name in "
        + repr(names)
        + "}}))"
    )
    record = json.loads(subprocess.check_output([python, "-c", code], text=True))
    record["compiler"] = compiler
    output = Path(args.environment_record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
