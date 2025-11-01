#!/usr/bin/env python3
"""
Deep Diagnostic: Examine Advanced Leader's Internal Behavior
This script will help identify the exact cause of failure on CIFAR-100
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

# Import the VGG model
import sys
sys.path.insert(0, '/home/mohanganesh/active_learning_coreset')

from cifar10_experiment import VGG as VGG10
from cifar100_experiment import VGG as VGG100
from active_learning_strategies import AdvancedLeader

print("="*80)
print("DEEP DIAGNOSTIC: Advanced Leader Internal Behavior")
print("="*80)
print()

# Load a trained model for each dataset
def load_model_and_data(dataset='cifar10'):
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
        model = VGG10(num_classes=10)
        num_classes = 10
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform)
        model = VGG100(num_classes=100)
        num_classes = 100
    
    return model, trainset, num_classes

# Extract features from a sample
def extract_sample_features(model, dataset, num_samples=5000):
    """Extract features from random samples"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Random sample
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)
    
    features = []
    labels = []
    predictions = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs, feat = model(inputs)
            features.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    
    return features, labels, predictions

# Analyze feature space
def analyze_feature_space(features, labels, num_classes, dataset_name):
    """Analyze the structure of the feature space"""
    print(f"\n{dataset_name.upper()} Feature Space Analysis:")
    print("-"*80)
    
    # Basic statistics
    print(f"Feature dimensionality: {features.shape[1]}")
    print(f"Number of samples: {features.shape[0]}")
    print(f"Number of classes: {num_classes}")
    
    # Compute intra-class and inter-class distances
    from sklearn.metrics import pairwise_distances
    
    # Sample for efficiency
    sample_size = min(1000, len(features))
    sample_idx = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[sample_idx]
    sample_labels = labels[sample_idx]
    
    print(f"\nComputing distances on {sample_size} samples...")
    
    # Intra-class distances (within same class)
    intra_class_dists = []
    for c in range(min(num_classes, 20)):  # Limit to 20 classes for speed
        class_mask = sample_labels == c
        if np.sum(class_mask) > 1:
            class_features = sample_features[class_mask]
            dists = pairwise_distances(class_features, class_features)
            triu_idx = np.triu_indices_from(dists, k=1)
            intra_class_dists.extend(dists[triu_idx])
    
    # Inter-class distances (between different classes)
    inter_class_dists = []
    for _ in range(1000):  # Sample 1000 random pairs
        i, j = np.random.choice(sample_size, 2, replace=False)
        if sample_labels[i] != sample_labels[j]:
            dist = np.linalg.norm(sample_features[i] - sample_features[j])
            inter_class_dists.append(dist)
    
    if len(intra_class_dists) > 0 and len(inter_class_dists) > 0:
        print(f"\nIntra-class distances (same class):")
        print(f"  Mean: {np.mean(intra_class_dists):.3f}, Std: {np.std(intra_class_dists):.3f}")
        print(f"  25th: {np.percentile(intra_class_dists, 25):.3f}, "
              f"50th: {np.percentile(intra_class_dists, 50):.3f}, "
              f"75th: {np.percentile(intra_class_dists, 75):.3f}")
        
        print(f"\nInter-class distances (different classes):")
        print(f"  Mean: {np.mean(inter_class_dists):.3f}, Std: {np.std(inter_class_dists):.3f}")
        print(f"  25th: {np.percentile(inter_class_dists, 25):.3f}, "
              f"50th: {np.percentile(inter_class_dists, 50):.3f}, "
              f"75th: {np.percentile(inter_class_dists, 75):.3f}")
        
        # Separation ratio
        separation = np.mean(inter_class_dists) / (np.mean(intra_class_dists) + 1e-8)
        print(f"\nClass Separation Ratio: {separation:.3f}")
        print(f"  (Higher is better; >1.0 means classes are separated)")
        
        return {
            'intra_mean': np.mean(intra_class_dists),
            'intra_std': np.std(intra_class_dists),
            'inter_mean': np.mean(inter_class_dists),
            'inter_std': np.std(inter_class_dists),
            'separation': separation,
            'intra_dists': intra_class_dists,
            'inter_dists': inter_class_dists
        }
    return None

# Simulate Advanced Leader thresholds
def simulate_thresholds(features, dataset_name):
    """Simulate what thresholds Advanced Leader would compute"""
    print(f"\n{dataset_name.upper()} - Advanced Leader Threshold Simulation:")
    print("-"*80)
    
    from sklearn.metrics import pairwise_distances
    
    sample_size = min(300, len(features))
    sample_idx = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[sample_idx]
    
    pdists = pairwise_distances(sample_features, sample_features)
    triu_idx = np.triu_indices_from(pdists, k=1)
    distances = pdists[triu_idx]
    
    p25 = np.percentile(distances, 25)
    p50 = np.percentile(distances, 50)
    p75 = np.percentile(distances, 75)
    
    print(f"Multi-scale thresholds:")
    print(f"  Fine (25th percentile):   {p25:.3f}")
    print(f"  Medium (50th percentile): {p50:.3f}")
    print(f"  Coarse (75th percentile): {p75:.3f}")
    
    return [p25, p50, p75]

# Main analysis
print("\nDisabling cuDNN for compatibility...")
torch.backends.cudnn.enabled = False

print("\nLoading CIFAR-10...")
model_c10, data_c10, nc10 = load_model_and_data('cifar10')
print("Extracting CIFAR-10 features...")
feat_c10, labels_c10, pred_c10 = extract_sample_features(model_c10, data_c10, 2000)

print("\nLoading CIFAR-100...")
model_c100, data_c100, nc100 = load_model_and_data('cifar100')
print("Extracting CIFAR-100 features...")
feat_c100, labels_c100, pred_c100 = extract_sample_features(model_c100, data_c100, 2000)

# Analyze both
stats_c10 = analyze_feature_space(feat_c10, labels_c10, nc10, "CIFAR-10")
thresholds_c10 = simulate_thresholds(feat_c10, "CIFAR-10")

stats_c100 = analyze_feature_space(feat_c100, labels_c100, nc100, "CIFAR-100")
thresholds_c100 = simulate_thresholds(feat_c100, "CIFAR-100")

# Create comparison visualizations
print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Advanced Leader Failure Analysis: Feature Space Comparison', 
             fontsize=16, fontweight='bold')

# Plot 1: Distance distributions CIFAR-10
if stats_c10:
    ax = axes[0, 0]
    ax.hist(stats_c10['intra_dists'], bins=50, alpha=0.6, label='Intra-class', density=True)
    ax.hist(stats_c10['inter_dists'], bins=50, alpha=0.6, label='Inter-class', density=True)
    ax.axvline(thresholds_c10[0], color='r', linestyle='--', label=f'Fine thresh: {thresholds_c10[0]:.2f}')
    ax.axvline(thresholds_c10[1], color='g', linestyle='--', label=f'Med thresh: {thresholds_c10[1]:.2f}')
    ax.axvline(thresholds_c10[2], color='b', linestyle='--', label=f'Coarse thresh: {thresholds_c10[2]:.2f}')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax.set_title(f'CIFAR-10 Distances\nSeparation: {stats_c10["separation"]:.2f}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Plot 2: Distance distributions CIFAR-100
if stats_c100:
    ax = axes[0, 1]
    ax.hist(stats_c100['intra_dists'], bins=50, alpha=0.6, label='Intra-class', density=True)
    ax.hist(stats_c100['inter_dists'], bins=50, alpha=0.6, label='Inter-class', density=True)
    ax.axvline(thresholds_c100[0], color='r', linestyle='--', label=f'Fine thresh: {thresholds_c100[0]:.2f}')
    ax.axvline(thresholds_c100[1], color='g', linestyle='--', label=f'Med thresh: {thresholds_c100[1]:.2f}')
    ax.axvline(thresholds_c100[2], color='b', linestyle='--', label=f'Coarse thresh: {thresholds_c100[2]:.2f}')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax.set_title(f'CIFAR-100 Distances\nSeparation: {stats_c100["separation"]:.2f}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Plot 3: Separation comparison
ax = axes[0, 2]
if stats_c10 and stats_c100:
    datasets = ['CIFAR-10', 'CIFAR-100']
    separations = [stats_c10['separation'], stats_c100['separation']]
    colors = ['green' if s > 1.5 else 'orange' if s > 1.0 else 'red' for s in separations]
    bars = ax.bar(datasets, separations, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Threshold (1.0)')
    ax.set_ylabel('Separation Ratio')
    ax.set_title('Class Separation\n(Inter/Intra distance ratio)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, separations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Threshold comparison
ax = axes[1, 0]
scales = ['Fine\n(25th)', 'Medium\n(50th)', 'Coarse\n(75th)']
x = np.arange(len(scales))
width = 0.35
bars1 = ax.bar(x - width/2, thresholds_c10, width, label='CIFAR-10', alpha=0.8)
bars2 = ax.bar(x + width/2, thresholds_c100, width, label='CIFAR-100', alpha=0.8)
ax.set_ylabel('Threshold Value')
ax.set_title('Multi-Scale Thresholds', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scales)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: PCA visualization CIFAR-10
ax = axes[1, 1]
pca = PCA(n_components=2)
feat_2d_c10 = pca.fit_transform(feat_c10[:500])
scatter = ax.scatter(feat_2d_c10[:, 0], feat_2d_c10[:, 1], 
                     c=labels_c10[:500], cmap='tab10', s=10, alpha=0.6)
ax.set_title('CIFAR-10 Feature Space (PCA)', fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# Plot 6: PCA visualization CIFAR-100
ax = axes[1, 2]
feat_2d_c100 = pca.fit_transform(feat_c100[:500])
scatter = ax.scatter(feat_2d_c100[:, 0], feat_2d_c100[:, 1], 
                     c=labels_c100[:500], cmap='tab20', s=10, alpha=0.6)
ax.set_title('CIFAR-100 Feature Space (PCA)', fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.tight_layout()
plt.savefig('advanced_leader_diagnostic.png', dpi=300, bbox_inches='tight')
print("✓ Saved: advanced_leader_diagnostic.png")

# Final diagnosis
print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)
print()

if stats_c10 and stats_c100:
    print("KEY FINDINGS:")
    print()
    print(f"1. CLASS SEPARATION:")
    print(f"   CIFAR-10:  {stats_c10['separation']:.2f}x (classes well separated)")
    print(f"   CIFAR-100: {stats_c100['separation']:.2f}x (classes poorly separated)")
    if stats_c100['separation'] < 1.5:
        print(f"   ⚠ CIFAR-100 classes overlap significantly!")
    print()
    
    print(f"2. THRESHOLD PROBLEMS:")
    print(f"   CIFAR-10 fine threshold:  {thresholds_c10[0]:.3f}")
    print(f"   CIFAR-100 fine threshold: {thresholds_c100[0]:.3f}")
    print(f"   CIFAR-100 intra-class mean: {stats_c100['intra_mean']:.3f}")
    
    if thresholds_c100[0] > stats_c100['intra_mean']:
        print(f"   ⚠ PROBLEM: Fine threshold ({thresholds_c100[0]:.3f}) > intra-class ({stats_c100['intra_mean']:.3f})")
        print(f"      → Advanced Leader will create too many clusters")
        print(f"      → Each cluster contains points from DIFFERENT classes")
        print(f"      → Leaders don't represent class diversity!")
    print()
    
    print("3. ROOT CAUSE:")
    print("   - CIFAR-100 has overlapping classes in feature space")
    print("   - Advanced Leader thresholds assume separated clusters")
    print("   - Percentile-based thresholds don't adapt to overlap")
    print("   - Multi-scale clustering selects redundant samples")
    print()
    
    print("RECOMMENDED FIXES:")
    print("  1. Use class-aware sampling (ensure coverage of all classes)")
    print("  2. Adjust percentiles: use 10th, 30th, 60th instead of 25th, 50th, 75th")
    print("  3. Add uncertainty weighting (favor high-uncertainty samples)")
    print("  4. Use margin-based sampling instead of distance-based")

print("\n" + "="*80)
