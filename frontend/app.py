from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR if (THIS_DIR / "data").exists() else THIS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PRODUCT_IMAGES_DIR = DATA_DIR / "product_images"
PRODUCTS_CSV = DATA_DIR / "product_dataset.csv"
CHECKPOINT_CANDIDATES = (PROJECT_ROOT / "frontend" / "best.pt", PROJECT_ROOT / "outputs" / "best.pt")
CHECKPOINT_PATH = next((path for path in CHECKPOINT_CANDIDATES if path.exists()), CHECKPOINT_CANDIDATES[0])
EMBEDDINGS_PT_CANDIDATES = (
    PROJECT_ROOT / "frontend" / "product_embeddings.pt",
    PROJECT_ROOT / "frontend" / "56.18" / "product_embeddings.pt",
)
EMBEDDINGS_TAR_PATH = PROJECT_ROOT / "frontend" / "56.18.tar.gz"
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
TOP_K = 8
BATCH_SIZE = 64

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


@app.after_request
def add_cors_headers(response):
    if request.path == "/predict-json" or request.path.startswith("/product-image/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def open_image_safe(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return None


class BundleRetriever:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model = self.model.to(self.device).eval()

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"No se encontro el checkpoint en {CHECKPOINT_PATH}")
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Formato de checkpoint invalido en {CHECKPOINT_PATH}")
        state_dict = strip_module_prefix(state_dict)
        self.model.load_state_dict(state_dict, strict=False)

        products_df = pd.read_csv(PRODUCTS_CSV)
        self.product_descriptions = {
            str(row.product_asset_id): str(row.product_description) if not pd.isna(row.product_description) else ""
            for row in products_df.itertuples(index=False)
        }
        self.product_image_map = {
            str(row.product_asset_id): PRODUCT_IMAGES_DIR / f"{row.product_asset_id}.jpg"
            for row in products_df.itertuples(index=False)
        }

        precomputed = self._load_precomputed_product_embeddings()
        if precomputed is None:
            product_ids = [pid for pid, path in self.product_image_map.items() if path.exists()]
            self.product_ids, self.product_embeddings = self._encode_product_index(product_ids)
            print("Embeddings: indexados en runtime")
        else:
            self.product_ids, self.product_embeddings = precomputed
            print("Embeddings: cargados precomputados")
        self.sim_device = self.product_embeddings.device
        print(f"Retriever listo en {self.device} | index en {self.sim_device}")

    def _load_precomputed_product_embeddings(self) -> Optional[tuple[List[str], torch.Tensor]]:
        payload = None

        for candidate in EMBEDDINGS_PT_CANDIDATES:
            if candidate.exists():
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                break

        if payload is None and EMBEDDINGS_TAR_PATH.exists():
            with tarfile.open(EMBEDDINGS_TAR_PATH, "r:gz") as archive:
                member = next(
                    (item for item in archive.getmembers() if item.isfile() and item.name.endswith("product_embeddings.pt")),
                    None,
                )
                if member is not None:
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        payload = torch.load(io.BytesIO(extracted.read()), map_location="cpu", weights_only=False)

        if payload is None or not isinstance(payload, dict):
            return None

        pids = payload.get("pids")
        embeddings = payload.get("embeddings")
        if not isinstance(pids, list) or not isinstance(embeddings, torch.Tensor):
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != len(pids):
            return None

        filtered_ids: List[str] = []
        filtered_indices: List[int] = []
        for idx, raw_pid in enumerate(pids):
            pid = str(raw_pid)
            image_path = self.product_image_map.get(pid)
            if image_path is not None and image_path.exists():
                filtered_ids.append(pid)
                filtered_indices.append(idx)

        if not filtered_ids:
            return None

        filtered_embeddings = embeddings[filtered_indices].float().contiguous()
        filtered_embeddings = F.normalize(filtered_embeddings, p=2, dim=1)
        filtered_embeddings = filtered_embeddings.to(self.device, non_blocking=True)
        return filtered_ids, filtered_embeddings

    @torch.inference_mode()
    def _encode_product_index(self, product_ids: List[str]) -> tuple[List[str], torch.Tensor]:
        encoded_ids: List[str] = []
        encoded_embs: List[torch.Tensor] = []

        for start in tqdm(range(0, len(product_ids), BATCH_SIZE), desc="Indexando productos"):
            chunk_ids = product_ids[start : start + BATCH_SIZE]
            batch_ids: List[str] = []
            batch_images: List[torch.Tensor] = []
            batch_texts: List[str] = []

            for product_id in chunk_ids:
                image = open_image_safe(self.product_image_map[product_id])
                if image is None:
                    continue
                batch_ids.append(product_id)
                batch_images.append(self.preprocess(image))
                batch_texts.append(self.product_descriptions.get(product_id, "").strip())

            if not batch_ids:
                continue

            images = torch.stack(batch_images, dim=0).to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                image_feats = self.model.encode_image(images).float()
                text_rows = [idx for idx, text in enumerate(batch_texts) if text]
                if text_rows:
                    texts = [batch_texts[idx] for idx in text_rows]
                    tokens = self.tokenizer(texts).to(self.device, non_blocking=True)
                    text_feats = self.model.encode_text(tokens).float()
                    image_feats[text_rows] = image_feats[text_rows] + text_feats
                feats = F.normalize(image_feats, p=2, dim=1)

            encoded_ids.extend(batch_ids)
            encoded_embs.append(feats)

        if not encoded_embs:
            raise RuntimeError("No se pudo indexar ningun producto.")

        embeddings = torch.cat(encoded_embs, dim=0).contiguous()
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device, non_blocking=True)
        return encoded_ids, embeddings

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = TOP_K) -> List[Dict[str, str]]:
        if not self.product_ids:
            return []

        query = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            query_feat = self.model.encode_image(query).float()
            query_feat = F.normalize(query_feat, p=2, dim=1)

        sims = query_feat @ self.product_embeddings.T
        k = min(top_k, len(self.product_ids))
        scores, indices = torch.topk(sims.squeeze(0), k=k, largest=True, sorted=True)

        results: List[Dict[str, str]] = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            product_id = self.product_ids[int(idx)]
            results.append(
                {
                    "product_asset_id": product_id,
                    "description": self.product_descriptions.get(product_id, ""),
                    "score": f"{float(score):.4f}",
                }
            )
        return results


retriever: Optional[BundleRetriever] = None
startup_error: Optional[str] = None


def get_retriever() -> Optional[BundleRetriever]:
    global retriever, startup_error
    if retriever is None and startup_error is None:
        try:
            retriever = BundleRetriever()
        except Exception as exc:  # pragma: no cover - startup/runtime safeguard
            startup_error = str(exc)
    return retriever


@app.get("/")
def index():
    return render_template("index.html", results=None, preview_image=None, error=startup_error)


@app.post("/predict")
def predict():
    model = get_retriever()
    if startup_error:
        return render_template("index.html", results=None, preview_image=None, error=startup_error), 500
    if model is None:
        return render_template("index.html", results=None, preview_image=None, error="No se pudo cargar el modelo"), 500

    file = request.files.get("bundle_image")
    if file is None or not file.filename:
        return render_template("index.html", results=None, preview_image=None, error="Sube una imagen primero"), 400

    raw = file.read()
    if not raw:
        return render_template("index.html", results=None, preview_image=None, error="La imagen esta vacia"), 400

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return render_template("index.html", results=None, preview_image=None, error="Archivo de imagen invalido"), 400

    preview_mime = file.mimetype if (file.mimetype or "").startswith("image/") else "image/jpeg"
    preview_image = f"data:{preview_mime};base64,{base64.b64encode(raw).decode('ascii')}"
    results = model.predict(image=image, top_k=TOP_K)

    return render_template("index.html", results=results, preview_image=preview_image, error=None)


@app.post("/predict-json")
def predict_json():
    model = get_retriever()
    if startup_error:
        return jsonify({"error": startup_error}), 500
    if model is None:
        return jsonify({"error": "No se pudo cargar el modelo"}), 500

    file = request.files.get("bundle_image")
    if file is None or not file.filename:
        return jsonify({"error": "Sube una imagen primero"}), 400

    raw = file.read()
    if not raw:
        return jsonify({"error": "La imagen esta vacia"}), 400

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return jsonify({"error": "Archivo de imagen invalido"}), 400

    results = model.predict(image=image, top_k=TOP_K)
    return jsonify({"results": results, "top_k": TOP_K, "checkpoint_path": str(CHECKPOINT_PATH)})


@app.get("/product-image/<product_id>")
def product_image(product_id: str):
    model = get_retriever()
    if model is None:
        abort(404)
    image_path = model.product_image_map.get(product_id)
    if image_path is None or not image_path.exists():
        abort(404)
    return send_file(image_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
