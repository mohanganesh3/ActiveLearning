#!/usr/bin/env python3
"""Debug script to understand why thresholds become 0.000"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def compute_multi_scale_thresholds_OLD(features):
    """OLD version that might have been used during experiments"""
    sample_size = min(300, len(features))
    if sample_size <= 1:
        return [0.5, 1.0, 1.5]

    sample_idx = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[sample_idx]

    try:
        # vectorized pairwise distances (upper triangle)
        pdists = pairwise_distances(sample_features, metric='euclidean')
        triu_idx = np.triu_indices_from(pdists, k=1)
        distances = pdists[triu_idx]
    except Exception:
        # fallback to small randomized pairs (cheap)
        distances = []
        pairs = min(2000, sample_size * 10)
        for _ in range(pairs):
            i = np.random.randint(0, sample_size)
            j = np.random.randint(0, sample_size)
            if i == j:
                continue
            distances.append(np.linalg.norm(sample_features[i] - sample_features[j]))

    if len(distances) == 0:
        return [0.5, 1.0, 1.5]

    base = float(np.median(distances))

    # If base is extremely small (near-zero), fallback to median k-NN distance
    if base < 1e-6:
        try:
            k = min(10, len(sample_features)-1)
            if k >= 1:
                nbrs = NearestNeighbors(n_neighbors=k).fit(sample_features)
                dists, _ = nbrs.kneighbors(sample_features)
                avg_dist = np.mean(dists, axis=1)
                base = float(np.median(avg_dist))
        except Exception:
            base = 0.5

    # OLD VERSION MIGHT NOT HAVE HAD THIS CHECK!
    # if base <= 0:
    #     base = 0.5

    return [base * 0.5, base * 1.0, base * 1.5]

def compute_multi_scale_thresholds_FIXED(features):
    """FIXED version with proper safety check"""
    sample_size = min(300, len(features))
    if sample_size <= 1:
        return [0.5, 1.0, 1.5]

    sample_idx = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[sample_idx]

    try:
        pdists = pairwise_distances(sample_features, metric='euclidean')
        triu_idx = np.triu_indices_from(pdists, k=1)
        distances = pdists[triu_idx]
    except Exception:
        distances = []
        pairs = min(2000, sample_size * 10)
        for _ in range(pairs):
            i = np.random.randint(0, sample_size)
            j = np.random.randint(0, sample_size)
            if i == j:
                continue
            distances.append(np.linalg.norm(sample_features[i] - sample_features[j]))

    if len(distances) == 0:
        return [0.5, 1.0, 1.5]

    base = float(np.median(distances))

    # If base is extremely small (near-zero), fallback to median k-NN distance
    if base < 1e-6:
        try:
            k = min(10, len(sample_features)-1)
            if k >= 1:
                nbrs = NearestNeighbors(n_neighbors=k).fit(sample_features)
                dists, _ = nbrs.kneighbors(sample_features)
                avg_dist = np.mean(dists, axis=1)
                base = float(np.median(avg_dist))
        except Exception:
            base = 0.5

    # FIXED: Ensure base > 0
    if base <= 0:
        base = 0.5

    return [base * 0.5, base * 1.0, base * 1.5]


print("="*80)
print("Testing with zero-variance features (all identical)")
print("="*80)
features_zero = np.ones((1000, 128)) * 0.5  # All features identical
print("Feature shape:", features_zero.shape)
print("Feature std:", np.std(features_zero))

old_thresholds = compute_multi_scale_thresholds_OLD(features_zero)
fixed_thresholds = compute_multi_scale_thresholds_FIXED(features_zero)

print(f"\nOLD version: {[f'{t:.6f}' for t in old_thresholds]}")
print(f"FIXED version: {[f'{t:.6f}' for t in fixed_thresholds]}")

print("\n" + "="*80)
print("Testing with extremely low-variance features (BatchNorm collapse)")
print("="*80)
# Simulate what happens with BatchNorm in eval mode with untrained model
# All features might be very similar but not exactly identical
features_low_var = np.ones((1000, 128)) * 0.5 + np.random.randn(1000, 128) * 1e-8
print("Feature shape:", features_low_var.shape)
print("Feature std:", np.std(features_low_var))

old_thresholds = compute_multi_scale_thresholds_OLD(features_low_var)
fixed_thresholds = compute_multi_scale_thresholds_FIXED(features_low_var)

print(f"\nOLD version: {[f'{t:.6f}' for t in old_thresholds]}")
print(f"FIXED version: {[f'{t:.6f}' for t in fixed_thresholds]}")

print("\n" + "="*80)
print("Testing with normal features")
print("="*80)
features_normal = np.random.randn(1000, 128)
print("Feature shape:", features_normal.shape)
print("Feature std:", np.std(features_normal))

old_thresholds = compute_multi_scale_thresholds_OLD(features_normal)
fixed_thresholds = compute_multi_scale_thresholds_FIXED(features_normal)

print(f"\nOLD version: {[f'{t:.6f}' for t in old_thresholds]}")
print(f"FIXED version: {[f'{t:.6f}' for t in fixed_thresholds]}")
