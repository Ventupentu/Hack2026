import argparse
import pandas as pd
from collections import defaultdict
from typing import List, Dict

def reciprocal_rank_fusion(csv_paths: List[str], top_k: int = 15, k_penalty: int = 60) -> pd.DataFrame:
    """
    Combines multiple submission CSVs using Reciprocal Rank Fusion (RRF).
    
    Args:
        csv_paths: List of paths to the submission CSV files.
        top_k: Number of top products to keep per bundle.
        k_penalty: Constant added to the rank to penalize lower ranked items (formula: 1 / (k_penalty + rank))
        
    Returns:
        A DataFrame with the ensembled predictions, containing 'bundle_asset_id' and 'product_asset_id'.
    """
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided.")
        
    print(f"Ensembling {len(csv_paths)} files using RRF (k={k_penalty})...")
    
    # bundle_id -> product_id -> rrf_score
    rrf_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    for path in csv_paths:
        print(f"  Reading: {path}")
        df = pd.read_csv(path)
        
        # We assume the CSV is ordered implicitly by score/rank since it's the submission format
        # Group by bundle to get the rank of each product
        for bundle_id, group in df.groupby('bundle_asset_id', sort=False):
            # The order in the group defines the rank (1-indexed)
            for rank, product_id in enumerate(group['product_asset_id'], start=1):
                # RRF Formula: 1 / (k + rank)
                score = 1.0 / (k_penalty + rank)
                rrf_scores[str(bundle_id)][str(product_id)] += score
                
    # Now, for each bundle, sort the products by their accumulated RRF score descending
    submission_rows = []
    
    # We must preserve the bundles from the first CSV (which contains all test bundles)
    bundles = pd.read_csv(csv_paths[0])['bundle_asset_id'].unique()
    
    for bundle_id in bundles:
        bundle_id_str = str(bundle_id)
        prod_scores = rrf_scores.get(bundle_id_str, {})
        
        # Sort products by score descending
        sorted_prods = sorted(prod_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take Top K
        top_prods = [prod_id for prod_id, score in sorted_prods[:top_k]]
        
        # Pad if for some reason we have less than top_k (unlikely in RRF of full CSVs)
        if len(top_prods) < top_k:
            print(f"Warning: Bundle {bundle_id} has less than {top_k} products after ensemble.")
            
        for prod_id in top_prods:
            submission_rows.append({
                'bundle_asset_id': bundle_id,
                'product_asset_id': prod_id
            })
            
    result_df = pd.DataFrame(submission_rows)
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple submission CSVs using RRF.")
    parser.add_argument("csvs", nargs="+", help="Paths to the CSV files to ensemble")
    parser.add_argument("--output", "-o", default="outputs/ensemble_submission.csv", help="Output file path")
    parser.add_argument("--top_k", "-k", type=int, default=15, help="Number of products per bundle")
    parser.add_argument("--penalty", "-p", type=int, default=60, help="RRF penalty constant")
    
    args = parser.parse_args()
    
    ensemble_df = reciprocal_rank_fusion(args.csvs, top_k=args.top_k, k_penalty=args.penalty)
    ensemble_df.to_csv(args.output, index=False)
    
    print(f"\nSaved ensembled submission to: {args.output}")
    print(f"Total rows: {len(ensemble_df)}")
    print(f"Bundles: {len(ensemble_df['bundle_asset_id'].unique())}")
    
    # Check padding
    counts = ensemble_df.groupby('bundle_asset_id').size()
    if not all(counts == args.top_k):
        print("WARNING: Some bundles do not have exactly 15 products!")
    else:
        print(f"Success: All bundles have exactly {args.top_k} products.")
