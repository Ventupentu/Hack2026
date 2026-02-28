"""Assign a gender/section label to every product in product_dataset.csv.

Strategy
--------
1. **Direct assignment** — If a product appears in ``bundles_product_match_train.csv``
   and all its associated bundles belong to a single ``bundle_id_section``, that
   section is assigned directly.  The analysis shows **zero** products span more
   than one section, so every linked product gets an unambiguous label.

2. **Heuristic inference for unlinked products** (~23 676 products have no bundle
   association).  We use the ``product_description`` field:
   a. Descriptions that start with ``BABY`` → section 3.
   b. Descriptions that are ≥90% exclusive to one section in the training data
      → that section (e.g. ``HEELED SHOES`` → 1, ``MOCCASINS`` → 2).
   c. Descriptions shared across sections (<90% exclusive) → 0
      (we do NOT guess — honest fallback).
   d. Descriptions never seen in training → 0 (fallback).

Output
------
Writes ``data/product_dataset_with_gender.csv`` with an extra column
``gender`` whose values are: ``1`` (mujer), ``2`` (hombre), ``3`` (kids),
``0`` (unisex / unknown).

Also prints a conflict report (should be 0) and a coverage summary.

Usage
-----
    python src/utils/add_gender.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

PRODUCTS_CSV = DATA_DIR / "product_dataset.csv"
TRAIN_CSV = DATA_DIR / "bundles_product_match_train.csv"
BUNDLES_CSV = DATA_DIR / "bundles_dataset.csv"
OUTPUT_CSV = DATA_DIR / "product_dataset_with_gender.csv"

# ---------------------------------------------------------------------------
# Section constants  (matches bundle_id_section in bundles_dataset.csv)
# ---------------------------------------------------------------------------

SECTION_MUJER = 1
SECTION_HOMBRE = 2
SECTION_KIDS = 3
SECTION_UNKNOWN = 0   # unisex / not assignable

# Descriptions that unambiguously belong to kids regardless of bundle data
BABY_PREFIXES = ("BABY ",)


def _is_baby_description(desc: str) -> bool:
    upper = desc.upper().strip()
    return any(upper.startswith(prefix) for prefix in BABY_PREFIXES)


def main() -> None:
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    products = pd.read_csv(PRODUCTS_CSV)
    train = pd.read_csv(TRAIN_CSV)
    bundles = pd.read_csv(BUNDLES_CSV)

    products["product_asset_id"] = products["product_asset_id"].astype(str)
    train["product_asset_id"] = train["product_asset_id"].astype(str)
    train["bundle_asset_id"] = train["bundle_asset_id"].astype(str)
    bundles["bundle_asset_id"] = bundles["bundle_asset_id"].astype(str)

    bundle_section = dict(
        zip(bundles["bundle_asset_id"], bundles["bundle_id_section"])
    )

    # ------------------------------------------------------------------
    # Step 1: Direct assignment from bundle links
    # ------------------------------------------------------------------
    product_sections: dict[str, set[int]] = defaultdict(set)
    for _, row in train.iterrows():
        pid = row["product_asset_id"]
        bid = row["bundle_asset_id"]
        section = bundle_section.get(bid)
        if section is not None:
            product_sections[pid].add(int(section))

    # Conflict report
    conflicts = {
        pid: secs for pid, secs in product_sections.items() if len(secs) > 1
    }
    print(f"Products with direct bundle link : {len(product_sections)}")
    print(f"Products in >1 section (conflict): {len(conflicts)}")
    if conflicts:
        for pid, secs in list(conflicts.items())[:10]:
            desc = products.loc[
                products["product_asset_id"] == pid, "product_description"
            ].values
            print(f"  CONFLICT {pid}: sections={secs}, desc={desc}")

    direct_gender: dict[str, int] = {}
    for pid, secs in product_sections.items():
        if len(secs) == 1:
            direct_gender[pid] = next(iter(secs))
        else:
            # Conflict resolution: majority vote from link counts
            section_counts: Counter[int] = Counter()
            for _, row in train[train["product_asset_id"] == pid].iterrows():
                sec = bundle_section.get(row["bundle_asset_id"])
                if sec is not None:
                    section_counts[int(sec)] += 1
            direct_gender[pid] = section_counts.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Step 2: Build description → section heuristic from training data
    # ------------------------------------------------------------------
    merged = train.merge(
        bundles[["bundle_asset_id", "bundle_id_section"]], on="bundle_asset_id"
    )
    merged = merged.merge(products, on="product_asset_id")

    desc_section_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for _, row in merged.iterrows():
        desc = str(row["product_description"]).strip().upper()
        desc_section_counts[desc][int(row["bundle_id_section"])] += 1

    # For each description: pick the section ONLY if ≥90% exclusive
    desc_to_section: dict[str, int] = {}
    for desc, counter in desc_section_counts.items():
        total = sum(counter.values())
        majority_section, majority_count = counter.most_common(1)[0]
        ratio = majority_count / total
        if ratio >= 0.90:
            desc_to_section[desc] = majority_section

    # ------------------------------------------------------------------
    # Step 3: Assign gender to every product
    # ------------------------------------------------------------------
    all_pids = products["product_asset_id"].tolist()
    all_descs = products["product_description"].fillna("").astype(str).tolist()

    gender_col: list[int] = []
    method_stats: Counter[str] = Counter()

    for pid, desc in zip(all_pids, all_descs):
        # Priority 1: direct bundle link
        if pid in direct_gender:
            gender_col.append(direct_gender[pid])
            method_stats["direct_link"] += 1
            continue

        desc_upper = desc.strip().upper()

        # Priority 2: BABY prefix → kids (3)
        if _is_baby_description(desc_upper):
            gender_col.append(SECTION_KIDS)
            method_stats["baby_prefix"] += 1
            continue

        # Priority 3: description heuristic from training distribution
        if desc_upper in desc_to_section:
            gender_col.append(desc_to_section[desc_upper])
            method_stats["desc_heuristic"] += 1
            continue

        # Priority 4: fallback → 0 (unknown)
        gender_col.append(SECTION_UNKNOWN)
        method_stats["fallback_unknown"] += 1

    products["gender"] = gender_col

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    LABELS = {0: "unknown/unisex", 1: "mujer", 2: "hombre", 3: "kids"}

    print()
    print("=== Assignment method distribution ===")
    for method, count in sorted(method_stats.items(), key=lambda x: -x[1]):
        print(f"  {method:20s}: {count:>6d} ({100*count/len(all_pids):.1f}%)")

    print()
    print("=== Gender distribution ===")
    for val, cnt in sorted(products["gender"].value_counts().items()):
        print(f"  {val} ({LABELS.get(val, '?'):15s}): {cnt}")

    print()
    print("=== Sample per gender ===")
    for g in sorted(products["gender"].unique()):
        sample = products[products["gender"] == g].head(3)[
            ["product_asset_id", "product_description", "gender"]
        ]
        print(f"--- {g} ({LABELS.get(g, '?')}) ---")
        print(sample.to_string(index=False))
        print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    products.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total products: {len(products)}")


if __name__ == "__main__":
    main()
