#!/usr/bin/env python3
"""
=============================================================================
VARIANTE 2: ARTICLE MATCH + ENSEMBLE REORDER
Score obtenido: 63.52 (mejor resultado, mejora sobre 60.28)
=============================================================================

Estrategia:
-----------
1. Partimos del mejor submission base (60.28).
2. Calculamos un score de confianza por producto usando Reciprocal Rank Fusion
   (RRF) ponderada sobre las 7 submissions disponibles.
3. Extraemos el artículo Zara de las URLs para hacer article matching.
4. Para cada bundle de test:
   a. Si existe un article match ausente → reemplaza el producto con menor
      score RRF (sin restricción de votos, a diferencia de V1).
   b. Reordenamos los 15 productos por score RRF, con los article matches
      en primera posición.

Justificación:
--------------
- Article matching: 93.1% de precisión validada en training (997/1071).
- RRF ponderada: combina la información de ranking de 7 submissions,
  dando más peso a las de mayor score. Esto permite:
    · Identificar los productos más confiables para reordenar.
    · Identificar los menos confiables para sustituir.
- El reordenamiento ayuda si la métrica de evaluación es sensible al orden
  (e.g., MAP, NDCG) o si se evalúan las primeras N filas con prioridad.

Parámetros RRF:
- K = 60 (constante estándar que suaviza diferencias de ranking)
- Pesos: w_i = exp((score_i - min_score) / 2) / Σ exp(...)
  Esto da ~37.8% de peso al 60.28 y ~4.9% al 56.18.

Resultado: 55 sustituciones + reordenamiento de todos los bundles.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# 1. CARGAR DATOS
# ============================================================
bundles = pd.read_csv('test/bundles_dataset.csv')
products = pd.read_csv('test/product_dataset.csv')
train = pd.read_csv('test/bundles_product_match_train.csv')
test = pd.read_csv('test/bundles_product_match_test.csv')

# ============================================================
# 2. EXTRAER NÚMERO DE ARTÍCULO ZARA DE LAS URLs
# ============================================================
def extract_article(url):
    """
    Extrae el número de artículo Zara de una URL de imagen.
    Ejemplo: .../02878302711-p/02878302711-p.jpg?ts=... → '02878302711'
    La referencia aparece en los últimos segmentos del path, antes de .jpg,
    y el sufijo (-p para bundles, -e1 para productos) se elimina.
    """
    if pd.isna(url):
        return None
    parts = url.split('/')
    for p in reversed(parts):
        if '.jpg' in p:
            return p.split('.jpg')[0].split('-')[0]
    return None

bundles['article'] = bundles['bundle_image_url'].apply(extract_article)
products['article'] = products['product_image_url'].apply(extract_article)

# Índice: artículo → [product_asset_id, ...]
prod_by_article = defaultdict(list)
for _, row in products.iterrows():
    if row['article']:
        prod_by_article[row['article']].append(row['product_asset_id'])

bundle_info = bundles.set_index('bundle_asset_id')

# ============================================================
# 3. CARGAR SUBMISSIONS
# ============================================================
submission_files = [
    (56.18, 'test_submission_phase1_56.18.csv'),
    (57.70, 'test_submission_phase1_57.70 .csv'),
    (57.83, 'test_submission_phase1_57.83.csv'),
    (57.90, 'test_submission_phase1_57.90.csv'),
    (57.96, 'test_submission_phase1_57.96.csv'),
    (58.03, 'test_submission_phase1_58.03.csv'),
    (60.28, 'test_submission_phase1_60.28 .csv'),
]

submissions = {}
for score, fname in submission_files:
    submissions[score] = pd.read_csv(fname)

best = submissions[60.28].copy()
test_bundles_list = test['bundle_asset_id'].unique()

# ============================================================
# 4. RECIPROCAL RANK FUSION (RRF) PONDERADA
# ============================================================
# Fórmula: score(p, b) = Σ_i  w_i / (K + rank_i(p, b))
#   - w_i: peso del submission i (proporcional a exp(score_i))
#   - rank_i: posición (1-15) del producto p en el submission i para el bundle b
#   - K: constante de suavizado (60, valor estándar en la literatura)
#   - Si p no aparece en el submission i, no aporta score.

K_RRF = 60

# Calcular pesos normalizados
scores_arr = np.array([s for s, _ in submission_files])
weights = np.exp((scores_arr - scores_arr.min()) / 2.0)
weights = weights / weights.sum()
weight_map = {s: w for s, w in zip(scores_arr, weights)}

print("Pesos RRF por submission:")
for s, w in zip(scores_arr, weights):
    print(f"  Score {s:.2f} → peso {w:.4f}")

# Calcular RRF score para cada (bundle, producto)
rrf_scores = {}  # {bundle_id: {product_id: rrf_score}}

for score_val, df in submissions.items():
    w = weight_map[score_val]
    for b_id in test_bundles_list:
        if b_id not in rrf_scores:
            rrf_scores[b_id] = defaultdict(float)
        bundle_prods = df[df['bundle_asset_id'] == b_id]['product_asset_id'].tolist()
        for rank, p_id in enumerate(bundle_prods, 1):
            rrf_scores[b_id][p_id] += w / (K_RRF + rank)

# ============================================================
# 5. GENERAR SUBMISSION: ARTICLE MATCH + REORDER
# ============================================================
rows = []
changes = 0

for b_id in test_bundles_list:
    # Productos actuales del 60.28
    current_prods = best[best['bundle_asset_id'] == b_id]['product_asset_id'].tolist()[:15]

    # Article match
    b_article = bundle_info.loc[b_id, 'article'] if b_id in bundle_info.index else None
    article_prods = []
    if b_article and b_article in prod_by_article:
        article_prods = prod_by_article[b_article]

    # Sustituir: si hay article match ausente, reemplazar el de menor RRF score
    new_prods = list(current_prods)
    for ap in article_prods:
        if ap not in new_prods:
            min_score = float('inf')
            min_idx = -1
            for i, p in enumerate(new_prods):
                s = rrf_scores[b_id].get(p, 0)
                if s < min_score:
                    min_score = s
                    min_idx = i
            if min_idx >= 0:
                new_prods[min_idx] = ap
                changes += 1

    # Reordenar por score RRF, con article matches al inicio
    article_set = set(article_prods)
    prod_scores = []
    for p in new_prods:
        s = rrf_scores[b_id].get(p, 0)
        # Los article matches reciben el score máximo para ir primero
        if p in article_set:
            s = max(s, 999)
        prod_scores.append((p, s))

    prod_scores.sort(key=lambda x: -x[1])

    for p, _ in prod_scores[:15]:
        rows.append({'bundle_asset_id': b_id, 'product_asset_id': p})

# ============================================================
# 6. GUARDAR RESULTADO
# ============================================================
result = pd.DataFrame(rows)
output_file = 'submission_v2_article_reorder.csv'
result.to_csv(output_file, index=False)

print(f"\nSubmission guardado: {output_file}")
print(f"Filas: {len(result)}, Bundles: {result['bundle_asset_id'].nunique()}")
print(f"Sustituciones por article match: {changes}")
print(f"Productos únicos: {result['product_asset_id'].nunique()}")
