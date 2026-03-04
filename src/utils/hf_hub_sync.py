"""Utilities for strict, asynchronous artifact sync to Hugging Face Hub."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

from src.config import InditexConfig


class HfArtifactUploader:
    """Background uploader for checkpoint bundles (checkpoint + metrics + config)."""

    def __init__(
        self,
        repo_id: str,
        token: str,
        artifact_namespace: str,
        hydra_config_path: Path,
    ) -> None:
        if not repo_id:
            raise ValueError("hf.hf_repo_id must be set when hf.push_to_hub=true")
        if not token:
            raise ValueError("hf.hf_token must be set when hf.push_to_hub=true")

        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "huggingface_hub is required when hf.push_to_hub=true. "
                "Install with: pip install huggingface_hub"
            ) from exc

        self._repo_id = repo_id
        self._token = token
        self._namespace = artifact_namespace.strip().strip("/")
        self._hydra_config_path = hydra_config_path
        self._api = HfApi(token=token)
        self._commit_operation_add = CommitOperationAdd
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-upload")
        self._futures: List[Future[None]] = []
        self._validate_startup()

    def _validate_startup(self) -> None:
        """Fail fast if token/repo are invalid."""
        try:
            self._api.whoami(token=self._token)
        except Exception as exc:
            raise RuntimeError(
                "Invalid Hugging Face token in hf.hf_token while hf.push_to_hub=true."
            ) from exc

        try:
            self._api.create_repo(
                repo_id=self._repo_id,
                repo_type="model",
                token=self._token,
                exist_ok=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to access or create Hugging Face model repo '{self._repo_id}'."
            ) from exc

    def _path_in_repo(self, filename: str) -> str:
        if self._namespace:
            return f"{self._namespace}/{filename}"
        return filename

    def _redacted_hydra_config_bytes(self) -> bytes:
        """Redact hf_token before pushing config to a remote repo."""
        raw = self._hydra_config_path.read_text(encoding="utf-8")
        redacted_lines: List[str] = []
        for line in raw.splitlines():
            stripped = line.lstrip()
            leading_ws = line[: len(line) - len(stripped)]
            if stripped.startswith("hf_token:"):
                redacted_lines.append(f'{leading_ws}hf_token: "***REDACTED***"')
            else:
                redacted_lines.append(line)
        redacted = "\n".join(redacted_lines)
        if raw.endswith("\n"):
            redacted += "\n"
        return redacted.encode("utf-8")

    def _upload_checkpoint_bundle(
        self,
        checkpoint_path: Path,
        metrics_path: Path,
        checkpoint_label: str,
    ) -> None:
        operations = [
            self._commit_operation_add(
                path_in_repo=self._path_in_repo(checkpoint_path.name),
                path_or_fileobj=str(checkpoint_path),
            ),
            self._commit_operation_add(
                path_in_repo=self._path_in_repo(metrics_path.name),
                path_or_fileobj=str(metrics_path),
            ),
            self._commit_operation_add(
                path_in_repo=self._path_in_repo("hydra_config.yaml"),
                path_or_fileobj=BytesIO(self._redacted_hydra_config_bytes()),
            ),
        ]
        self._api.create_commit(
            repo_id=self._repo_id,
            repo_type="model",
            token=self._token,
            operations=operations,
            commit_message=f"Upload {self._namespace or 'training'} {checkpoint_label} artifacts",
        )

    def _raise_completed_failures(self) -> None:
        pending: List[Future[None]] = []
        for future in self._futures:
            if not future.done():
                pending.append(future)
                continue
            error = future.exception()
            if error is not None:
                raise RuntimeError("Background Hugging Face upload failed.") from error
        self._futures = pending

    def queue_checkpoint_artifacts(
        self,
        checkpoint_path: Path,
        metrics_path: Path,
        checkpoint_label: str,
    ) -> None:
        self._raise_completed_failures()
        required_paths = [checkpoint_path, metrics_path, self._hydra_config_path]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Cannot upload artifacts because required files are missing: "
                + ", ".join(missing_paths)
            )

        future = self._executor.submit(
            self._upload_checkpoint_bundle,
            checkpoint_path.resolve(),
            metrics_path.resolve(),
            checkpoint_label,
        )
        self._futures.append(future)

    def wait_for_pending_uploads(self) -> None:
        if self._futures:
            wait(self._futures)
        self._raise_completed_failures()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


def _read_hf_value(cfg_obj: Any, key: str, default: Any) -> Any:
    if cfg_obj is None:
        return default
    value = getattr(cfg_obj, key, default)
    return default if value is None else value


def build_hf_uploader(
    cfg: InditexConfig,
    output_dir: Path,
    artifact_namespace: str,
) -> Optional[HfArtifactUploader]:
    """Create uploader only when hf.push_to_hub=true."""
    hf_cfg = getattr(cfg, "hf", None)
    push_to_hub = bool(_read_hf_value(hf_cfg, "push_to_hub", False))
    if not push_to_hub:
        return None

    repo_id = str(_read_hf_value(hf_cfg, "hf_repo_id", "")).strip()
    token = str(_read_hf_value(hf_cfg, "hf_token", "")).strip()
    hydra_config_path = (output_dir.parent / ".hydra" / "config.yaml").resolve()
    if not hydra_config_path.exists():
        raise FileNotFoundError(
            f"Hydra runtime config not found at {hydra_config_path}. "
            "Expected this file to upload training config to Hugging Face Hub."
        )

    return HfArtifactUploader(
        repo_id=repo_id,
        token=token,
        artifact_namespace=artifact_namespace,
        hydra_config_path=hydra_config_path,
    )
