#!/usr/bin/env python3
"""
=============================================================================
VARIANTE 1: CONSERVADORA — Article Match Fix
Score obtenido: 63.38 (mejora sobre 60.28)
=============================================================================

Estrategia:
-----------
1. Partimos del mejor submission base (60.28).
2. Extraemos el número de artículo Zara de las URLs de bundles y productos.
   - Las URLs de bundle contienen una referencia como '02878302711-p'
   - Las URLs de producto contienen una referencia como '02878302711-e1'
   - Cuando coinciden, es casi seguro que el producto pertenece al bundle.
3. Para cada bundle de test:
   a. Si existe un producto cuyo artículo coincide con el del bundle y NO está
      en la predicción actual del 60.28 → lo añadimos.
   b. Para hacer espacio, eliminamos el producto con menor número de "votos"
      entre las 7 submissions disponibles (solo si tiene ≤4 votos de 7).

Justificación:
--------------
- Validado en training: el article matching tiene 93.1% de precisión
  (997 TP vs 74 FP sobre 1071 bundles con match).
- El submission 60.28 no incluía 55 de estos article matches → oportunidad clara.
- Al eliminar solo productos con ≤4 votos (minoría), minimizamos el riesgo
  de quitar un producto correcto.

Resultado: 53 sustituciones quirúrgicas sobre 6825 predicciones.
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
    Ejemplo URL: .../02878302711-p/02878302711-p.jpg?ts=...
    Retorna: '02878302711'
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

# Índice de búsqueda: artículo → lista de product_asset_id
prod_by_article = defaultdict(list)
for _, row in products.iterrows():
    if row['article']:
        prod_by_article[row['article']].append(row['product_asset_id'])

bundle_info = bundles.set_index('bundle_asset_id')

# ============================================================
# 3. CARGAR LAS 7 SUBMISSIONS Y CONTAR VOTOS POR PRODUCTO
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

# Base: mejor submission
best = submissions[60.28].copy()
test_bundles_list = test['bundle_asset_id'].unique()

# Votos: para cada (bundle, producto), cuántas submissions lo incluyen
vote_counts = {}  # {bundle_id: Counter({product_id: nº votos})}
for score_val, df in submissions.items():
    for b_id in test_bundles_list:
        if b_id not in vote_counts:
            vote_counts[b_id] = Counter()
        bundle_prods = df[df['bundle_asset_id'] == b_id]['product_asset_id'].tolist()
        for p_id in bundle_prods:
            vote_counts[b_id][p_id] += 1

# ============================================================
# 4. GENERAR SUBMISSION: APLICAR ARTICLE MATCH FIX
# ============================================================
rows = []
changes = 0

for b_id in test_bundles_list:
    # Productos actuales del submission 60.28
    current_prods = best[best['bundle_asset_id'] == b_id]['product_asset_id'].tolist()[:15]

    # Buscar productos con article match para este bundle
    b_article = bundle_info.loc[b_id, 'article'] if b_id in bundle_info.index else None
    article_prods = []
    if b_article and b_article in prod_by_article:
        article_prods = prod_by_article[b_article]

    # Si hay un article match que falta en la predicción → sustituir
    new_prods = list(current_prods)
    for ap in article_prods:
        if ap not in new_prods:
            # Encontrar el producto con menor nº de votos para reemplazar
            min_votes = float('inf')
            min_idx = -1
            for i, p in enumerate(new_prods):
                v = vote_counts[b_id].get(p, 0)
                if v < min_votes:
                    min_votes = v
                    min_idx = i
            # Solo reemplazar si el producto a quitar tiene ≤4 votos (minoría)
            if min_votes <= 4 and min_idx >= 0:
                new_prods[min_idx] = ap
                changes += 1

    for p in new_prods[:15]:
        rows.append({'bundle_asset_id': b_id, 'product_asset_id': p})

# ============================================================
# 5. GUARDAR RESULTADO
# ============================================================
result = pd.DataFrame(rows)
output_file = 'submission_v1_conservative.csv'
result.to_csv(output_file, index=False)

print(f"Submission guardado: {output_file}")
print(f"Filas: {len(result)}, Bundles: {result['bundle_asset_id'].nunique()}")
print(f"Sustituciones realizadas: {changes}")
print(f"Productos únicos: {result['product_asset_id'].nunique()}")
