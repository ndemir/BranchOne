"""Command-line entry points; help and parsing do not import model libraries."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="branchone")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare",
        help="Prepare raw Choice JSONL and reject duplicates/overlength inputs",
    )
    prepare.add_argument("--split", action="append", required=True, metavar="NAME=PATH")
    prepare.add_argument("--image-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--workers", type=int, default=4)
    train = sub.add_parser(
        "train", help="Start a new stage or exactly resume an interrupted stage"
    )
    train.add_argument(
        "--train", required=True, help="Prepared training JSONL or JSONL.GZ"
    )
    train.add_argument(
        "--development", required=True, help="Prepared fixed development-panel JSONL"
    )
    train.add_argument("--image-root", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config")
    mode = train.add_mutually_exclusive_group()
    mode.add_argument(
        "--init-checkpoint",
        help="Existing model weights; start a fresh optimizer/schedule",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Restore this output stage's complete training state",
    )
    train.add_argument("--expected-training-rows", type=int)
    train.add_argument(
        "--max-updates",
        type=int,
        help="Interrupt cleanly after this many new updates; does not shorten the stage",
    )
    predict = sub.add_parser(
        "predict", help="Reference text/image inference from a model checkpoint"
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument(
        "--input",
        required=True,
        help="Request JSON; image path is local to this machine",
    )
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--output")
    evaluation = sub.add_parser(
        "evaluate",
        help="Evaluate frozen weights or fit temperature on a calibration split",
    )
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--image-root", required=True)
    evaluation.add_argument("--output", required=True)
    calibration = evaluation.add_mutually_exclusive_group()
    calibration.add_argument("--calibration")
    calibration.add_argument("--fit-calibration", action="store_true")
    args = parser.parse_args(argv)
    # Set threading before importing PyTorch/NumPy in a fresh interpreter.
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ.setdefault(name, "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "prepare":
        from .prepare import prepare_data

        splits = {}
        for value in args.split:
            name, separator, path = value.partition("=")
            if not separator or not path:
                parser.error("--split must have the form NAME=PATH")
            splits.setdefault(name, []).append(path)
        prepare_data(splits, args.image_root, args.output, workers=args.workers)
        return 0
    if args.command == "train":
        from .stage import run_stage

        return run_stage(
            train_path=args.train,
            development_path=args.development,
            image_root=args.image_root,
            output=args.output,
            config_path=args.config,
            init_checkpoint=args.init_checkpoint,
            resume=args.resume,
            expected_training_rows=args.expected_training_rows,
            max_updates=args.max_updates,
        )
    if args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(
            args.checkpoint,
            args.data,
            args.image_root,
            args.output,
            calibration=args.calibration,
            fit_calibration=args.fit_calibration,
        )
        return 0
    from .io import write_json
    from .model import BranchOne

    request = json.loads(Path(args.input).read_text())
    model = BranchOne.load(args.checkpoint, device=args.device)
    result = model.predict(request)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
