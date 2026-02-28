from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def save_product_embeddings(path: str | Path, product_ids: list[str], embeddings: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, product_ids=np.asarray(product_ids), embeddings=embeddings.astype(np.float32))


def load_product_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(Path(path), allow_pickle=False)
    product_ids = [str(x) for x in payload["product_ids"].tolist()]
    embeddings = payload["embeddings"].astype(np.float32)
    return product_ids, embeddings


class ProductIndex:
    """Product embedding index with brute-force or FAISS search."""

    def __init__(self, mode: str = "brute", use_gpu: bool = False, device: str | None = None) -> None:
        if mode not in {"brute", "faiss"}:
            raise ValueError(f"Unsupported index mode: {mode}")
        self.mode = mode
        self.use_gpu = use_gpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.product_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._tensor_index: torch.Tensor | None = None
        self._faiss_index = None

    def build(self, product_ids: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be rank-2 [N, D]")
        if len(product_ids) != embeddings.shape[0]:
            raise ValueError("product_ids length must match embeddings")

        self.product_ids = product_ids
        self.embeddings = embeddings.astype(np.float32)

        if self.mode == "brute":
            self._tensor_index = torch.from_numpy(self.embeddings).to(self.device)
            LOGGER.info("Built brute index with %d vectors on %s", len(product_ids), self.device)
            return

        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faiss is required when mode='faiss'") from exc

        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        if self.use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("Installed faiss package has no GPU support")
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        index.add(self.embeddings)
        self._faiss_index = index
        LOGGER.info("Built faiss index with %d vectors (gpu=%s)", len(product_ids), self.use_gpu)

    def search(self, query_embeddings: np.ndarray | torch.Tensor, topk: int = 200) -> tuple[np.ndarray, np.ndarray]:
        target_device = torch.device(self.device)
        if isinstance(query_embeddings, torch.Tensor):
            q = query_embeddings
            if q.device != target_device:
                q = q.to(target_device)
            q_np = q.detach().cpu().numpy().astype(np.float32)
        else:
            q_np = query_embeddings.astype(np.float32)
            q = torch.from_numpy(q_np).to(target_device)

        if q_np.ndim == 1:
            q_np = q_np[None, :]
            q = q[None, :]

        if self.mode == "brute":
            if self._tensor_index is None:
                raise RuntimeError("Index not built")
            sim = torch.matmul(q, self._tensor_index.T)
            scores, indices = torch.topk(sim, k=min(topk, sim.shape[1]), dim=1)
            return scores.detach().cpu().numpy(), indices.detach().cpu().numpy()

        if self._faiss_index is None:
            raise RuntimeError("Index not built")
        scores, indices = self._faiss_index.search(q_np, min(topk, len(self.product_ids)))
        return scores.astype(np.float32), indices.astype(np.int64)

    def save(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None:
            raise RuntimeError("Cannot save empty index")

        save_product_embeddings(out_path / "product_embeddings.npz", self.product_ids, self.embeddings)
        meta = {
            "mode": self.mode,
            "use_gpu": self.use_gpu,
            "num_products": len(self.product_ids),
            "dim": int(self.embeddings.shape[1]),
        }
        (out_path / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self.mode == "faiss":
            import faiss

            index = self._faiss_index
            if index is None:
                raise RuntimeError("Cannot save unbuilt faiss index")
            if self.use_gpu:
                index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, str(out_path / "faiss.index"))

    @classmethod
    def load(cls, out_dir: str | Path, device: str | None = None) -> "ProductIndex":
        out_path = Path(out_dir)
        meta = json.loads((out_path / "index_meta.json").read_text(encoding="utf-8"))

        index = cls(mode=meta["mode"], use_gpu=bool(meta.get("use_gpu", False)), device=device)
        product_ids, embeddings = load_product_embeddings(out_path / "product_embeddings.npz")
        index.product_ids = product_ids
        index.embeddings = embeddings

        if index.mode == "brute":
            index._tensor_index = torch.from_numpy(embeddings).to(index.device)
            return index

        import faiss

        faiss_index = faiss.read_index(str(out_path / "faiss.index"))
        if index.use_gpu:
            resources = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(resources, 0, faiss_index)
        index._faiss_index = faiss_index
        return index
