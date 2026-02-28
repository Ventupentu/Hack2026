#!/usr/bin/env python3
"""Generate an interactive HTML gallery that loads bundles from an uploaded CSV."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd


PRODUCT_ID_COLUMN = "product_asset_id"
PRODUCT_DESCRIPTION_COLUMN = "product_description"
HARDCODED_BUNDLE_IMAGES_DIR = Path("data/bundle_images")
HARDCODED_PRODUCT_IMAGES_DIR = Path("data/product_images")
HARDCODED_PRODUCTS_CSV = Path("data/product_dataset.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive HTML file that lets you choose a CSV "
            "from the browser and renders the bundle/product gallery."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frontend/outputs/bundle_gallery_interactive.html"),
    )
    parser.add_argument(
        "--title",
        default="Bundle and Product Gallery",
        help="HTML page title.",
    )
    return parser.parse_args()


def to_relative_url(target: Path, output_parent: Path) -> str:
    rel_path = os.path.relpath(target, start=output_parent)
    return quote(rel_path.replace(os.sep, "/"), safe="/._-")


def load_product_descriptions(products_csv: Path) -> dict[str, str]:
    df = pd.read_csv(
        products_csv, usecols=[PRODUCT_ID_COLUMN, PRODUCT_DESCRIPTION_COLUMN]
    )
    clean = df.dropna(subset=[PRODUCT_ID_COLUMN]).copy()
    clean[PRODUCT_ID_COLUMN] = clean[PRODUCT_ID_COLUMN].astype(str).str.strip()
    clean[PRODUCT_DESCRIPTION_COLUMN] = (
        clean[PRODUCT_DESCRIPTION_COLUMN].fillna("").astype(str).str.strip()
    )
    return {
        pid: description
        for pid, description in zip(
            clean[PRODUCT_ID_COLUMN], clean[PRODUCT_DESCRIPTION_COLUMN]
        )
        if pid
    }


def generate_html(
    title: str,
    bundle_image_base: str,
    product_image_base: str,
    product_descriptions: dict[str, str],
) -> str:
    product_descriptions_json = json.dumps(
        product_descriptions,
        ensure_ascii=True,
        separators=(",", ":"),
    )

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="es">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #222; }",
        "h1 { margin-bottom: 8px; }",
        ".subtitle { color: #555; margin-bottom: 14px; }",
        ".controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }",
        "#csv-input { display: none; }",
        "#load-btn { border: 1px solid #0a0a0a; border-radius: 999px; background: linear-gradient(145deg, #1f2937, #0f172a); color: #fff; font-size: 0.95rem; font-weight: 700; padding: 10px 20px; cursor: pointer; transition: transform 0.15s ease, box-shadow 0.2s ease; box-shadow: 0 8px 18px rgba(2, 6, 23, 0.24); }",
        "#load-btn:hover { transform: translateY(-1px); box-shadow: 0 12px 24px rgba(2, 6, 23, 0.32); }",
        "#status { color: #555; font-size: 13px; }",
        ".bundle { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 20px; }",
        ".bundle-header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }",
        ".bundle-image { width: 220px; max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }",
        ".products { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; }",
        ".product-card { border: 1px solid #ececec; border-radius: 8px; padding: 8px; background: #fafafa; }",
        ".product-image { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 6px; border: 1px solid #ddd; }",
        ".asset-id { font-size: 12px; margin-top: 6px; word-break: break-all; }",
        ".meta { color: #555; font-size: 13px; margin-top: 4px; }",
        ".missing { display: flex; align-items: center; justify-content: center; background: #f3f3f3; color: #666; min-height: 120px; padding: 10px; text-align: center; }",
        "@media (max-width: 700px) { body { margin: 12px; } .bundle-image { width: 100%; max-width: 320px; } }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        '<p class="subtitle">Sube un CSV con columnas <code>bundle_asset_id</code> y <code>product_asset_id</code>.</p>',
        '<div class="controls">',
        '<input id="csv-input" type="file" accept=".csv,text/csv">',
        '<button id="load-btn" type="button">Seleccionar CSV</button>',
        '<span id="status">No hay archivo cargado.</span>',
        "</div>",
        '<div id="summary"></div>',
        '<div id="gallery"></div>',
        "<script>",
        f"const PRODUCT_DESCRIPTIONS = {product_descriptions_json};",
        f"const BUNDLE_IMAGE_BASE = {json.dumps(bundle_image_base)};",
        f"const PRODUCT_IMAGE_BASE = {json.dumps(product_image_base)};",
        "",
        "const csvInput = document.getElementById('csv-input');",
        "const loadBtn = document.getElementById('load-btn');",
        "const statusEl = document.getElementById('status');",
        "const summaryEl = document.getElementById('summary');",
        "const galleryEl = document.getElementById('gallery');",
        "",
        "function parseCsv(text) {",
        "  const rows = [];",
        "  let row = [];",
        "  let field = '';",
        "  let inQuotes = false;",
        "",
        "  for (let i = 0; i < text.length; i += 1) {",
        "    const ch = text[i];",
        "    if (inQuotes) {",
        "      if (ch === '\"') {",
        "        if (text[i + 1] === '\"') {",
        "          field += '\"';",
        "          i += 1;",
        "        } else {",
        "          inQuotes = false;",
        "        }",
        "      } else {",
        "        field += ch;",
        "      }",
        "      continue;",
        "    }",
        "",
        "    if (ch === '\"') {",
        "      inQuotes = true;",
        "      continue;",
        "    }",
        "    if (ch === ',') {",
        "      row.push(field);",
        "      field = '';",
        "      continue;",
        "    }",
        "    if (ch === '\\n') {",
        "      row.push(field);",
        "      rows.push(row);",
        "      row = [];",
        "      field = '';",
        "      continue;",
        "    }",
        "    if (ch === '\\r') {",
        "      continue;",
        "    }",
        "    field += ch;",
        "  }",
        "",
        "  if (field.length > 0 || row.length > 0) {",
        "    row.push(field);",
        "    rows.push(row);",
        "  }",
        "",
        "  return rows.filter((cells) => cells.some((cell) => cell.trim() !== ''));",
        "}",
        "",
        "function parseBundlesFromCsv(csvText) {",
        "  const rows = parseCsv(csvText);",
        "  if (rows.length === 0) {",
        "    throw new Error('El CSV esta vacio.');",
        "  }",
        "",
        "  const header = rows[0].map((cell) => cell.trim());",
        "  if (header.length > 0) {",
        "    header[0] = header[0].replace(/^\\uFEFF/, '');",
        "  }",
        "",
        "  const bundleIdx = header.indexOf('bundle_asset_id');",
        "  const productIdx = header.indexOf('product_asset_id');",
        "",
        "  if (bundleIdx === -1 || productIdx === -1) {",
        "    throw new Error('El CSV debe incluir columnas bundle_asset_id y product_asset_id.');",
        "  }",
        "",
        "  const grouped = new Map();",
        "  const seenPairs = new Set();",
        "",
        "  for (let i = 1; i < rows.length; i += 1) {",
        "    const cells = rows[i];",
        "    const bundleId = String(cells[bundleIdx] || '').trim();",
        "    const productId = String(cells[productIdx] || '').trim();",
        "    if (!bundleId || !productId) {",
        "      continue;",
        "    }",
        "",
        "    const pairKey = bundleId + '\\u0000' + productId;",
        "    if (seenPairs.has(pairKey)) {",
        "      continue;",
        "    }",
        "    seenPairs.add(pairKey);",
        "",
        "    if (!grouped.has(bundleId)) {",
        "      grouped.set(bundleId, []);",
        "    }",
        "    grouped.get(bundleId).push(productId);",
        "  }",
        "",
        "  return Array.from(grouped.entries());",
        "}",
        "",
        "function createMissingNode(cssClass, assetId) {",
        "  const missing = document.createElement('div');",
        "  missing.className = cssClass + ' missing';",
        "  missing.textContent = 'Missing image: ' + assetId;",
        "  return missing;",
        "}",
        "",
        "function createImageNode(basePath, assetId, cssClass) {",
        "  const img = document.createElement('img');",
        "  img.className = cssClass;",
        "  img.src = basePath + '/' + encodeURIComponent(assetId) + '.jpg';",
        "  img.alt = assetId;",
        "  img.loading = 'lazy';",
        "  img.referrerPolicy = 'no-referrer';",
        "  img.addEventListener(",
        "    'error',",
        "    () => {",
        "      img.replaceWith(createMissingNode(cssClass, assetId));",
        "    },",
        "    { once: true },",
        "  );",
        "  return img;",
        "}",
        "",
        "function renderGallery(bundles) {",
        "  galleryEl.replaceChildren();",
        "  summaryEl.replaceChildren();",
        "",
        "  const summary = document.createElement('p');",
        "  summary.textContent = 'Total bundles shown: ' + bundles.length;",
        "  summaryEl.appendChild(summary);",
        "",
        "  for (const [bundleId, productIds] of bundles) {",
        "    const section = document.createElement('section');",
        "    section.className = 'bundle';",
        "",
        "    const header = document.createElement('div');",
        "    header.className = 'bundle-header';",
        "",
        "    header.appendChild(createImageNode(BUNDLE_IMAGE_BASE, bundleId, 'bundle-image'));",
        "",
        "    const titleWrap = document.createElement('div');",
        "    const h2 = document.createElement('h2');",
        "    h2.textContent = 'Bundle ' + bundleId;",
        "    const count = document.createElement('div');",
        "    count.textContent = 'Products in bundle: ' + productIds.length;",
        "    titleWrap.appendChild(h2);",
        "    titleWrap.appendChild(count);",
        "    header.appendChild(titleWrap);",
        "",
        "    const productsGrid = document.createElement('div');",
        "    productsGrid.className = 'products';",
        "",
        "    for (const productId of productIds) {",
        "      const card = document.createElement('article');",
        "      card.className = 'product-card';",
        "",
        "      card.appendChild(createImageNode(PRODUCT_IMAGE_BASE, productId, 'product-image'));",
        "",
        "      const asset = document.createElement('div');",
        "      asset.className = 'asset-id';",
        "      asset.textContent = productId;",
        "",
        "      const meta = document.createElement('div');",
        "      meta.className = 'meta';",
        "      const description = PRODUCT_DESCRIPTIONS[productId] || '(not found)';",
        "      meta.textContent = 'product_description: ' + description;",
        "",
        "      card.appendChild(asset);",
        "      card.appendChild(meta);",
        "      productsGrid.appendChild(card);",
        "    }",
        "",
        "    section.appendChild(header);",
        "    section.appendChild(productsGrid);",
        "    galleryEl.appendChild(section);",
        "  }",
        "}",
        "",
        "function handleCsvSelection() {",
        "  const file = csvInput.files && csvInput.files[0];",
        "  if (!file) {",
        "    return;",
        "  }",
        "",
        "  statusEl.textContent = 'Procesando ' + file.name + '...';",
        "  const reader = new FileReader();",
        "",
        "  reader.onload = () => {",
        "    try {",
        "      const csvText = String(reader.result || '');",
        "      const bundles = parseBundlesFromCsv(csvText);",
        "      renderGallery(bundles);",
        "      statusEl.textContent = 'Archivo cargado: ' + file.name + ' | bundles: ' + bundles.length;",
        "    } catch (error) {",
        "      galleryEl.replaceChildren();",
        "      summaryEl.replaceChildren();",
        "      statusEl.textContent = error instanceof Error ? error.message : 'Error al procesar el CSV.';",
        "    }",
        "  };",
        "",
        "  reader.onerror = () => {",
        "    statusEl.textContent = 'No se pudo leer el archivo seleccionado.';",
        "  };",
        "",
        "  reader.readAsText(file, 'utf-8');",
        "}",
        "",
        "loadBtn.addEventListener('click', () => csvInput.click());",
        "csvInput.addEventListener('change', handleCsvSelection);",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def main() -> None:
    args = parse_args()

    if not HARDCODED_PRODUCTS_CSV.exists():
        raise FileNotFoundError(
            f"No se encontro el CSV hardcodeado de productos: {HARDCODED_PRODUCTS_CSV}"
        )

    output_parent = args.output.parent.resolve()
    bundle_image_base = to_relative_url(
        target=HARDCODED_BUNDLE_IMAGES_DIR.resolve(),
        output_parent=output_parent,
    )
    product_image_base = to_relative_url(
        target=HARDCODED_PRODUCT_IMAGES_DIR.resolve(),
        output_parent=output_parent,
    )
    product_descriptions = load_product_descriptions(HARDCODED_PRODUCTS_CSV)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_html(
        title=args.title,
        bundle_image_base=bundle_image_base,
        product_image_base=product_image_base,
        product_descriptions=product_descriptions,
    )
    args.output.write_text(html_content, encoding="utf-8")

    print(f"HTML written to: {args.output}")
    print(f"Hardcoded bundle images: {HARDCODED_BUNDLE_IMAGES_DIR}")
    print(f"Hardcoded product images: {HARDCODED_PRODUCT_IMAGES_DIR}")
    print(f"Hardcoded product categories: {HARDCODED_PRODUCTS_CSV}")


if __name__ == "__main__":
    main()
