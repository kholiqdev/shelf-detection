#!/usr/bin/env python3
"""Train Shelf Detection YOLO model from the command line.

This script mirrors the training flow from the notebook, but is designed for
long-running CLI sessions, for example inside GNU screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def project_root_from(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    return current


def parse_args() -> argparse.Namespace:
    default_project_root = project_root_from(Path.cwd())

    parser = argparse.ArgumentParser(
        description="Train YOLOv8m Shelf Detection on the official SKU-110K dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument("--persistent-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)

    parser.add_argument("--model-name", default="yolov8m.pt")
    parser.add_argument("--dataset-name", default="SKU-110K.yaml")
    parser.add_argument("--run-name", default="shelf-detection-cli-v1")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--save-dataset-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-dataset-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)
    parser.add_argument("--local-dataset-dir", type=Path, default=None)

    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--wandb-project", default="shelfscan")
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="offline",
        help="Use online only when WANDB_API_KEY is available.",
    )

    return parser.parse_args()


def require_dependencies() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    try:
        import torch
    except ModuleNotFoundError:
        missing.append("torch")
        torch = None
    try:
        import ultralytics
        from ultralytics import YOLO
    except ModuleNotFoundError:
        missing.append("ultralytics")
        ultralytics = None
        YOLO = None
    try:
        import wandb
    except ModuleNotFoundError:
        missing.append("wandb")
        wandb = None

    if missing:
        packages = ", ".join(missing)
        raise SystemExit(
            f"Missing Python package(s): {packages}\n\n"
            "Install for A100/NVIDIA CUDA with:\n"
            "  python -m pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            "  python -m pip install 'ultralytics>=8.4.70' 'wandb>=0.18.0'\n"
        )

    return torch, ultralytics, YOLO, wandb


def git_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def is_sku110k_prepared(path: Path) -> bool:
    return path.is_dir() and all((path / f"{split}.txt").is_file() for split in EXPECTED_SPLITS)


def write_sku110k_yaml(yaml_path: Path, dataset_path: Path) -> Path:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {dataset_path}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        "  0: object\n",
        encoding="utf-8",
    )
    return yaml_path


def sync_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        subprocess.run(
            [rsync, "-ah", "--info=progress2", f"{source}/", f"{destination}/"],
            check=True,
        )
        return

    shutil.copytree(source, destination, dirs_exist_ok=True)


def resolve_split_path(split_source: str | Path, dataset_root: Path) -> Path:
    split_path = Path(split_source)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    return split_path


def count_split_images(split_value: Any, dataset_root: Path) -> int:
    split_sources = split_value if isinstance(split_value, list) else [split_value]
    image_count = 0

    for split_source in split_sources:
        split_path = resolve_split_path(split_source, dataset_root)
        if split_path.is_file():
            image_count += sum(1 for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif split_path.is_dir():
            image_count += sum(1 for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise FileNotFoundError(f"Split source not found: {split_path}")

    return image_count


def prepare_dataset(args: argparse.Namespace) -> tuple[str, Path, Path, dict[str, dict[str, int]]]:
    from ultralytics.data.utils import check_det_dataset

    persistent_root = args.persistent_root
    runtime_root = args.runtime_root
    cache_dir = args.dataset_cache_dir
    local_dataset_dir = args.local_dataset_dir
    local_dataset_yaml = runtime_root / "SKU-110K.local.yaml"
    persistent_dataset_yaml = persistent_root / "datasets" / "SKU-110K.drive.yaml"

    dataset_for_training = args.dataset_name

    if args.restore_dataset_cache and is_sku110k_prepared(cache_dir):
        if not is_sku110k_prepared(local_dataset_dir):
            print(f"Restoring SKU-110K cache to runtime: {local_dataset_dir}", flush=True)
            sync_directory(cache_dir, local_dataset_dir)
        dataset_for_training = str(write_sku110k_yaml(local_dataset_yaml, local_dataset_dir))
    else:
        print("SKU-110K cache not found. Ultralytics will prepare the official descriptor.", flush=True)

    dataset_info = check_det_dataset(dataset_for_training)
    dataset_root = Path(dataset_info["path"])
    dataset_counts: dict[str, dict[str, int]] = {}

    for split in EXPECTED_SPLITS:
        split_value = dataset_info.get(split)
        if not split_value:
            raise RuntimeError(f"Split {split} is missing from {dataset_for_training}.")

        image_count = count_split_images(split_value, dataset_root)
        if image_count == 0:
            raise RuntimeError(f"Split {split} is empty.")
        dataset_counts[split] = {"images": image_count}
        print(f"{split:>5}: {image_count:>6} images", flush=True)

    if args.save_dataset_cache and not is_sku110k_prepared(cache_dir):
        print(f"Saving SKU-110K cache to persistent storage: {cache_dir}", flush=True)
        sync_directory(dataset_root, cache_dir)
    else:
        print("Dataset cache backup skipped or already available.", flush=True)

    write_sku110k_yaml(persistent_dataset_yaml, cache_dir)
    print(f"Dataset root active : {dataset_root}", flush=True)
    print(f"Dataset for training: {dataset_for_training}", flush=True)
    print(f"Persistent YAML     : {persistent_dataset_yaml}", flush=True)

    return dataset_for_training, dataset_root, persistent_dataset_yaml, dataset_counts


def setup_wandb(args: argparse.Namespace, wandb: Any, git_sha: str) -> Any | None:
    os.environ["WANDB_DIR"] = str(args.persistent_root / "wandb")
    os.environ["WANDB_PROJECT"] = args.wandb_project
    (args.persistent_root / "wandb").mkdir(parents=True, exist_ok=True)

    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B disabled.", flush=True)
        return None

    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required when --wandb-mode online is used.")

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb_config = {
        "model": args.model_name,
        "data": args.dataset_name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "git_commit": git_sha,
    }
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=wandb_config,
        mode=args.wandb_mode,
    )
    print(f"W&B mode: {args.wandb_mode}. Logs: {os.environ['WANDB_DIR']}", flush=True)
    return run


def training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "device": args.device,
        "workers": args.workers,
        "amp": args.amp,
        "cache": args.cache,
        "project": str(args.runs_dir),
        "name": args.run_name,
        "exist_ok": args.exist_ok,
        "verbose": True,
    }


def locate_best_model(args: argparse.Namespace, train_results: Any) -> Path:
    save_dir = Path(getattr(train_results, "save_dir", ""))
    best_from_result = save_dir / "weights" / "best.pt"
    if best_from_result.is_file():
        return best_from_result

    candidates = list(args.runs_dir.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"best.pt not found under {args.runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def save_artifacts(
    args: argparse.Namespace,
    best_source: Path,
    git_sha: str,
    dataset_for_training: str,
    dataset_root: Path,
    persistent_dataset_yaml: Path,
    training_metrics: dict[str, float],
    dataset_counts: dict[str, dict[str, int]],
    config: dict[str, Any],
) -> None:
    version_dir = args.models_dir / args.run_name
    version_dir.mkdir(parents=True, exist_ok=True)

    versioned_model = version_dir / "best.pt"
    latest_model = args.models_dir / "best.pt"
    shutil.copy2(best_source, versioned_model)
    shutil.copy2(best_source, latest_model)

    sha256 = hashlib.sha256(versioned_model.read_bytes()).hexdigest()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "git_commit": git_sha,
        "model_name": args.model_name,
        "model_source": str(best_source),
        "model_path": str(versioned_model),
        "latest_model_path": str(latest_model),
        "sha256": sha256,
        "training_config": {"model": args.model_name, "data": dataset_for_training, **config},
        "dataset_name": args.dataset_name,
        "dataset_for_training": dataset_for_training,
        "dataset_root": str(dataset_root),
        "dataset_cache_root": str(args.dataset_cache_dir),
        "dataset_cache_yaml": str(persistent_dataset_yaml),
        "training_metrics": training_metrics,
        "dataset_counts": dataset_counts,
    }
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model version : {versioned_model}", flush=True)
    print(f"Model latest  : {latest_model}", flush=True)
    print(f"Metadata      : {metadata_path}", flush=True)
    print(f"SHA-256       : {sha256}", flush=True)


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.persistent_root = (args.persistent_root or args.project_root).resolve()
    args.runtime_root = (args.runtime_root or args.project_root).resolve()
    args.runs_dir = (args.runs_dir or args.persistent_root / "runs").resolve()
    args.models_dir = (args.models_dir or args.persistent_root / "models").resolve()
    args.dataset_cache_dir = (args.dataset_cache_dir or args.persistent_root / "datasets" / "SKU-110K").resolve()
    args.local_dataset_dir = (args.local_dataset_dir or args.runtime_root / "SKU-110K").resolve()

    for directory in (args.persistent_root, args.runs_dir, args.models_dir, args.persistent_root / "datasets"):
        directory.mkdir(parents=True, exist_ok=True)

    torch, ultralytics, YOLO, wandb = require_dependencies()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Use --device cpu --allow-cpu only for intentional CPU smoke tests."
        )
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU training is disabled by default. Pass --allow-cpu if this is intentional.")

    git_sha = git_commit(args.project_root)
    print(f"Project root : {args.project_root}", flush=True)
    print(f"Persistent   : {args.persistent_root}", flush=True)
    print(f"Runtime      : {args.runtime_root}", flush=True)
    print(f"Git commit   : {git_sha}", flush=True)
    print(f"PyTorch      : {torch.__version__}", flush=True)
    print(f"Ultralytics  : {ultralytics.__version__}", flush=True)
    print(f"Device       : {args.device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA         : {torch.version.cuda}", flush=True)

    config = training_config(args)
    print(json.dumps({"model": args.model_name, "data": args.dataset_name, **config}, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete. Training was not started.", flush=True)
        return 0

    wandb_run = setup_wandb(args, wandb, git_sha)
    dataset_for_training, dataset_root, persistent_dataset_yaml, dataset_counts = prepare_dataset(args)

    if args.resume_checkpoint:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
        print(f"Resuming training from {args.resume_checkpoint}", flush=True)
        train_results = YOLO(str(args.resume_checkpoint)).train(resume=True)
    else:
        print("Starting new Ultralytics training run.", flush=True)
        train_results = YOLO(args.model_name).train(data=dataset_for_training, **config)

    training_metrics = {
        "mAP50": float(train_results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(train_results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(train_results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(train_results.results_dict.get("metrics/recall(B)", 0)),
    }
    print(json.dumps(training_metrics, indent=2), flush=True)

    if wandb_run is not None:
        wandb.log(training_metrics)
        wandb_run.summary.update(training_metrics)
        wandb.finish()

    best_source = locate_best_model(args, train_results)
    save_artifacts(
        args=args,
        best_source=best_source,
        git_sha=git_sha,
        dataset_for_training=dataset_for_training,
        dataset_root=dataset_root,
        persistent_dataset_yaml=persistent_dataset_yaml,
        training_metrics=training_metrics,
        dataset_counts=dataset_counts,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
