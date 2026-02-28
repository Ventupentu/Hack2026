"""Hugging Face Hub sync helpers with reproducibility and traceability metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "run"


def _safe_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def _find_git_root(start: Path) -> Optional[Path]:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(args: Sequence[str], cwd: Optional[Path]) -> str:
    if cwd is None:
        return ""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def collect_git_info(start_path: Path) -> Dict[str, Any]:
    root = _find_git_root(start_path)
    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    dirty_output = _run_git(["status", "--porcelain"], root)
    return {
        "git_root": str(root) if root is not None else "",
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(dirty_output),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_file_record(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    stat = path.stat()
    record.update(
        {
            "size_bytes": int(stat.st_size),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": sha256_file(path),
        }
    )
    return record


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    return {"raw_cfg": str(cfg)}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class HFHubSync:
    """Synchronize model/inference artifacts to HF Hub with trace metadata."""

    def __init__(
        self,
        *,
        enabled: bool,
        stage: str,
        artifact_root: Path,
        repo_id: str = "",
        private: bool = False,
        token_env: str = "HF_TOKEN",
        fail_on_error: bool = False,
        create_pr: bool = False,
        run_name: str = "",
        remote_train_dir: str = "artifacts/train",
        remote_infer_dir: str = "artifacts/infer",
        license_name: str = "apache-2.0",
        pipeline_tag: str = "image-feature-extraction",
        base_model: str = "Marqo/marqo-fashionSigLIP",
        datasets: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.stage = str(stage).strip().lower()
        self.artifact_root = artifact_root.resolve()
        self.repo_id = str(repo_id).strip()
        self.private = bool(private)
        self.token_env = str(token_env).strip() or "HF_TOKEN"
        self.fail_on_error = bool(fail_on_error)
        self.create_pr = bool(create_pr)
        self.license_name = str(license_name).strip() or "apache-2.0"
        self.pipeline_tag = str(pipeline_tag).strip() or "image-feature-extraction"
        self.base_model = str(base_model).strip() or "Marqo/marqo-fashionSigLIP"
        self.datasets = [str(x) for x in (datasets or []) if str(x).strip()]
        self.tags = [str(x) for x in (tags or []) if str(x).strip()]
        self.remote_train_dir = str(remote_train_dir).strip().strip("/") or "artifacts/train"
        self.remote_infer_dir = str(remote_infer_dir).strip().strip("/") or "artifacts/infer"
        self.trace_dir = (self.artifact_root / "traceability").resolve()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._run_context: Dict[str, Any] = {}

        git_info = collect_git_info(self.artifact_root)
        short_commit = (git_info.get("commit") or "")[:8] or "nogit"
        raw_run_name = run_name.strip() or f"{_utc_now_compact()}-{short_commit}"
        self.run_id = _slugify(raw_run_name)
        self.remote_root = self.remote_train_dir if self.stage == "train" else self.remote_infer_dir

        self._token = os.getenv(self.token_env) or None
        self._api: Any = None
        self._commit_add_cls: Any = None
        self._repo_ready = False

        if self.enabled:
            if not self.repo_id:
                raise ValueError("hub.enabled=true requires hub.repo_id to be set.")
            try:
                from huggingface_hub import CommitOperationAdd, HfApi

                self._api = HfApi()
                self._commit_add_cls = CommitOperationAdd
            except Exception as exc:
                raise RuntimeError(
                    "huggingface_hub is required for hub sync. Install with: pip install huggingface_hub"
                ) from exc

    @classmethod
    def from_config(cls, cfg: Any, artifact_root: Path, stage: str) -> "HFHubSync":
        hub_cfg = getattr(cfg, "hub", None)
        if hub_cfg is None or not bool(getattr(hub_cfg, "enabled", False)):
            return cls(enabled=False, stage=stage, artifact_root=artifact_root)
        return cls(
            enabled=True,
            stage=stage,
            artifact_root=artifact_root,
            repo_id=str(getattr(hub_cfg, "repo_id", "")),
            private=bool(getattr(hub_cfg, "private", False)),
            token_env=str(getattr(hub_cfg, "token_env", "HF_TOKEN")),
            fail_on_error=bool(getattr(hub_cfg, "fail_on_error", False)),
            create_pr=bool(getattr(hub_cfg, "create_pr", False)),
            run_name=str(getattr(hub_cfg, "run_name", "")),
            remote_train_dir=str(getattr(hub_cfg, "remote_train_dir", "artifacts/train")),
            remote_infer_dir=str(getattr(hub_cfg, "remote_infer_dir", "artifacts/infer")),
            license_name=str(getattr(hub_cfg, "license", "apache-2.0")),
            pipeline_tag=str(getattr(hub_cfg, "pipeline_tag", "image-feature-extraction")),
            base_model=str(getattr(hub_cfg, "base_model", "Marqo/marqo-fashionSigLIP")),
            datasets=list(getattr(hub_cfg, "datasets", []) or []),
            tags=list(getattr(hub_cfg, "tags", []) or []),
        )

    def _handle_error(self, action: str, exc: Exception) -> None:
        message = f"HF Hub sync failed during '{action}': {exc}"
        if self.fail_on_error:
            raise RuntimeError(message) from exc
        print(f"Warning: {message}")

    def _remote_path(self, relative: str) -> str:
        relative = relative.strip().strip("/")
        chunks = [self.remote_root, self.run_id]
        if relative:
            chunks.append(relative)
        return "/".join(chunks)

    def _ensure_repo(self) -> bool:
        if not self.enabled:
            return False
        if self._repo_ready:
            return True
        try:
            self._api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
                token=self._token,
            )
            self._repo_ready = True
            return True
        except Exception as exc:
            self._handle_error("create_repo", exc)
            return False

    def _create_commit(self, files: Mapping[str, Path], commit_message: str) -> None:
        if not self.enabled:
            return
        if not self._ensure_repo():
            return
        operations = []
        for path_in_repo, local_path in files.items():
            path = Path(local_path)
            if not path.exists():
                continue
            operations.append(
                self._commit_add_cls(path_in_repo=path_in_repo.strip("/"), path_or_fileobj=str(path))
            )
        if not operations:
            return
        try:
            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type="model",
                operations=operations,
                commit_message=commit_message,
                token=self._token,
                create_pr=self.create_pr,
            )
        except Exception as exc:
            self._handle_error("create_commit", exc)

    def _env_snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": _utc_now_iso(),
            "stage": self.stage,
            "run_id": self.run_id,
            "repo_id": self.repo_id,
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "token_env": self.token_env,
            "token_available": bool(self._token),
            "versions": {
                "torch": _safe_version("torch"),
                "open_clip_torch": _safe_version("open_clip_torch"),
                "numpy": _safe_version("numpy"),
                "pandas": _safe_version("pandas"),
                "hydra-core": _safe_version("hydra-core"),
                "huggingface_hub": _safe_version("huggingface_hub"),
            },
            "git": collect_git_info(self.artifact_root),
        }

    def _render_model_card(self, best_metric_name: str, best_metric_value: float, best_epoch: int) -> str:
        lines = [
            "---",
            f"license: {self.license_name}",
            f"pipeline_tag: {self.pipeline_tag}",
            "library_name: open_clip",
            f"base_model: {self.base_model}",
        ]
        if self.tags:
            lines.append("tags:")
            lines.extend([f"- {tag}" for tag in self.tags])
        if self.datasets:
            lines.append("datasets:")
            lines.extend([f"- {dataset}" for dataset in self.datasets])
        lines.append("---")
        lines.extend(
            [
                "",
                "# Fashion Bundle Retrieval (Open-Source)",
                "",
                "This repository is synchronized automatically from training and inference runs.",
                "",
                "## Latest Training Snapshot",
                f"- run_id: `{self.run_id}`",
                f"- best epoch: `{best_epoch}`",
                f"- {best_metric_name}: `{best_metric_value:.6f}`",
                f"- generated_utc: `{_utc_now_iso()}`",
                "",
                "## Traceability",
                "- Full run metadata and file hashes are versioned under:",
                f"  - `{self.remote_root}/{self.run_id}/traceability/`",
                "- Latest best checkpoint is mirrored at `checkpoints/best.pt`.",
                "",
                "## Reproducibility",
                "- Training/inference configs are exported as resolved config snapshots.",
                "- Dataset and artifact SHA256 digests are logged for every sync event.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def start_run(
        self,
        *,
        cfg: Any,
        data_files: Optional[Mapping[str, Path]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        context = {
            **self._env_snapshot(),
            "config": _cfg_to_dict(cfg),
            "data_files": {
                key: build_file_record(Path(path))
                for key, path in (data_files or {}).items()
            },
            "extra": dict(extra or {}),
        }
        self._run_context = context
        run_context_path = self.trace_dir / "run_context.json"
        config_path = self.trace_dir / "resolved_config.json"
        _write_json(run_context_path, context)
        _write_json(config_path, context["config"])

        if not self.enabled:
            return
        files = {
            self._remote_path("traceability/run_context.json"): run_context_path,
            self._remote_path("traceability/resolved_config.json"): config_path,
        }
        self._create_commit(files, commit_message=f"[{self.stage}] start run {self.run_id}")

    def publish_train_event(
        self,
        *,
        epoch: int,
        metric_name: str,
        metric_value: float,
        metrics_path: Path,
        checkpoint_path: Optional[Path],
        is_best: bool,
        push_every_epoch: bool,
        push_best: bool,
    ) -> None:
        if not push_every_epoch and not (is_best and push_best):
            return

        payload: Dict[str, Any] = {
            "event": "train_epoch",
            "timestamp_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "epoch": int(epoch),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "is_best": bool(is_best),
            "metrics_file": build_file_record(metrics_path),
        }
        if checkpoint_path is not None:
            payload["checkpoint_file"] = build_file_record(checkpoint_path)
        if self._run_context:
            payload["git"] = self._run_context.get("git", {})

        event_name = f"epoch_{epoch:03d}_{'best' if is_best else 'regular'}.json"
        event_path = self.trace_dir / event_name
        _write_json(event_path, payload)

        files: Dict[str, Path] = {
            self._remote_path(f"traceability/{event_name}"): event_path,
            self._remote_path("metrics/metrics.jsonl"): metrics_path,
        }
        if checkpoint_path is not None and checkpoint_path.exists():
            files[self._remote_path(f"checkpoints/{checkpoint_path.name}")] = checkpoint_path

        if is_best and push_best and checkpoint_path is not None and checkpoint_path.exists():
            latest_path = self.trace_dir / "latest_train.json"
            latest_payload = {
                "timestamp_utc": _utc_now_iso(),
                "run_id": self.run_id,
                "best_epoch": int(epoch),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "best_checkpoint": build_file_record(checkpoint_path),
            }
            _write_json(latest_path, latest_payload)
            card_path = self.trace_dir / "README.md"
            card_path.write_text(
                self._render_model_card(metric_name, float(metric_value), int(epoch)),
                encoding="utf-8",
            )
            files["checkpoints/best.pt"] = checkpoint_path
            files["latest_train.json"] = latest_path
            files["README.md"] = card_path

        if not self.enabled:
            return
        commit_msg = (
            f"[train] epoch={epoch} {metric_name}={metric_value:.6f}"
            + (" (best)" if is_best else "")
        )
        self._create_commit(files, commit_message=commit_msg)

    def publish_train_complete(
        self,
        *,
        best_epoch: int,
        best_metric_name: str,
        best_metric_value: float,
        metrics_path: Path,
        best_checkpoint_path: Optional[Path],
    ) -> None:
        payload: Dict[str, Any] = {
            "event": "train_complete",
            "timestamp_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "best_epoch": int(best_epoch),
            "best_metric_name": best_metric_name,
            "best_metric_value": float(best_metric_value),
            "metrics_file": build_file_record(metrics_path),
        }
        if best_checkpoint_path is not None:
            payload["best_checkpoint"] = build_file_record(best_checkpoint_path)
        end_path = self.trace_dir / "train_complete.json"
        _write_json(end_path, payload)
        if not self.enabled:
            return
        files = {self._remote_path("traceability/train_complete.json"): end_path}
        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            files[self._remote_path("checkpoints/best.pt")] = best_checkpoint_path
            files["checkpoints/best.pt"] = best_checkpoint_path
        self._create_commit(
            files,
            commit_message=(
                f"[train] complete best_{best_metric_name}={float(best_metric_value):.6f}"
            ),
        )

    def publish_inference(
        self,
        *,
        cfg: Any,
        submission_path: Path,
        metrics_path: Path,
        checkpoint_path: Optional[Path] = None,
        summary: Optional[Mapping[str, Any]] = None,
        data_files: Optional[Mapping[str, Path]] = None,
        push_inference: bool = True,
    ) -> None:
        payload: Dict[str, Any] = {
            "event": "inference",
            "timestamp_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "config": _cfg_to_dict(cfg),
            "summary": dict(summary or {}),
            "submission": build_file_record(submission_path),
            "metrics": build_file_record(metrics_path),
            "data_files": {
                key: build_file_record(Path(path))
                for key, path in (data_files or {}).items()
            },
            "git": collect_git_info(self.artifact_root),
        }
        if checkpoint_path is not None:
            payload["checkpoint"] = build_file_record(checkpoint_path)

        stamp = _utc_now_compact()
        trace_name = f"inference_{stamp}.json"
        trace_path = self.trace_dir / trace_name
        _write_json(trace_path, payload)

        if not self.enabled or not push_inference:
            return
        files: Dict[str, Path] = {
            self._remote_path(f"traceability/{trace_name}"): trace_path,
        }
        if submission_path.exists():
            files[self._remote_path(f"outputs/{submission_path.name}")] = submission_path
        if metrics_path.exists():
            files[self._remote_path(f"outputs/{metrics_path.name}")] = metrics_path

        latest_path = self.trace_dir / "latest_inference.json"
        _write_json(
            latest_path,
            {
                "timestamp_utc": _utc_now_iso(),
                "run_id": self.run_id,
                "submission": build_file_record(submission_path),
                "metrics": build_file_record(metrics_path),
            },
        )
        files["latest_inference.json"] = latest_path
        self._create_commit(files, commit_message=f"[inference] publish {self.run_id}")
