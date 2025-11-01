#!/usr/bin/env python3
"""
IMPROVED Advanced Leader Clustering - Adaptive version for both CIFAR-10 and CIFAR-100

Key improvements:
1. Adaptive percentiles based on number of classes (detected from predictions)
2. Adaptive k for density estimation
3. Class-aware diversity scoring
4. Stratified leader selection
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class ImprovedAdvancedLeader:
    """
    Improved Advanced Leader with adaptive parameters
    
    Automatically adapts to:
    - Fine-grained problems (100+ classes) → lower percentiles, higher k
    - Coarse-grained problems (≤20 classes) → original parameters
    """
    def __init__(self, N, budget):
        self.N = N
        self.budget = budget
        self.num_classes = None  # Will be detected from predictions
    
    def select_batch(self, model, unlabeled_data):
        model.eval()
        device = next(model.parameters()).device
        
        # Extract features, uncertainties, AND predictions
        print("   [Improved] Extracting features + uncertainties + predictions...")
        features, uncertainties, predictions = self._extract_features_and_uncertainty(
            model, unlabeled_data, device
        )
        num_samples = len(features)
        
        # Detect number of classes from predictions
        self.num_classes = len(np.unique(predictions))
        print(f"   [Improved] Detected {self.num_classes} classes from predictions")
        
        # Adaptive k based on number of classes
        adaptive_k = self._get_adaptive_k(self.num_classes)
        print(f"   [Improved] Using adaptive k={adaptive_k} for density estimation")
        
        # Compute local densities using adaptive k
        densities = self._compute_densities(features, k=adaptive_k)
        
        # Compute adaptive multi-scale thresholds
        thresholds = self._compute_adaptive_thresholds(features, self.num_classes)
        print(f"   [Improved] Adaptive thresholds: {[f'{t:.3f}' for t in thresholds]}")
        
        # Class-aware multi-scale clustering
        print(f"   [Improved] Class-aware multi-scale clustering ({num_samples} points)...")
        candidate_leaders = self._class_aware_clustering(
            features, uncertainties, densities, predictions, thresholds
        )
        
        print(f"   [Improved] Candidate leaders: {len(candidate_leaders)}")
        
        # Score and select best candidates with class diversity
        if len(candidate_leaders) > self.budget:
            selected = self._score_and_select_diverse(
                candidate_leaders, features, uncertainties, densities, predictions
            )
        else:
            selected = candidate_leaders
        
        # Fill remaining budget with stratified uncertainty sampling
        if len(selected) < self.budget:
            selected = self._fill_with_stratified_uncertainty(
                selected, predictions, uncertainties, num_samples
            )
        
        print(f"   [Improved] Final selection: {len(selected)} samples")
        return selected[:self.budget]
    
    def _get_adaptive_k(self, num_classes):
        """Adaptive k for density estimation based on number of classes"""
        if num_classes <= 20:
            return 10  # Original for coarse-grained
        elif num_classes <= 50:
            return 20  # Medium-grained
        else:
            return 30  # Fine-grained (CIFAR-100)
    
    def _extract_features_and_uncertainty(self, model, data, device):
        """Extract features, uncertainty, AND predictions"""
        features = []
        uncertainties = []
        predictions = []
        loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc="   Features", leave=False):
                inputs = inputs.to(device)
                outputs, feat = model(inputs)
                
                # Features
                features.append(feat.cpu().numpy())
                
                # Uncertainty: entropy of predictions
                probs = F.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainties.append(entropy.cpu().numpy())
                
                # Predictions (pseudo-labels)
                preds = torch.argmax(outputs, dim=1)
                predictions.append(preds.cpu().numpy())
        
        return (np.concatenate(features, axis=0), 
                np.concatenate(uncertainties, axis=0),
                np.concatenate(predictions, axis=0))
    
    def _compute_densities(self, features, k=10):
        """Compute local density using k-nearest neighbors"""
        if len(features) < k:
            return np.ones(len(features))
        
        nbrs = NearestNeighbors(n_neighbors=min(k, len(features)-1)).fit(features)
        distances, _ = nbrs.kneighbors(features)
        
        avg_dist = np.mean(distances, axis=1)
        densities = 1.0 / (avg_dist + 1e-8)
        return densities
    
    def _compute_adaptive_thresholds(self, features, num_classes):
        """
        Adaptive percentiles based on number of classes
        
        Key insight: Fine-grained problems need LOWER percentiles
        """
        sample_size = min(300, len(features))
        if sample_size <= 1:
            return [0.5, 1.0, 1.5]

        sample_idx = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features[sample_idx]

        try:
            from sklearn.metrics import pairwise_distances
            pdists = pairwise_distances(sample_features, metric='euclidean')
            triu_idx = np.triu_indices_from(pdists, k=1)
            distances = pdists[triu_idx]
        except Exception:
            distances = []
            pairs = min(2000, sample_size * 10)
            for _ in range(pairs):
                i = np.random.randint(0, sample_size)
                j = np.random.randint(0, sample_size)
                if i != j:
                    distances.append(np.linalg.norm(sample_features[i] - sample_features[j]))

        if len(distances) == 0:
            return [0.5, 1.0, 1.5]

        distances = np.array(distances)
        
        # ADAPTIVE PERCENTILES based on problem complexity
        if num_classes <= 20:
            # Coarse-grained (CIFAR-10): Original percentiles
            percentiles = [25, 50, 75]
            print(f"   [Adaptive] Using coarse-grained percentiles: {percentiles}")
        elif num_classes <= 50:
            # Medium-grained: Slightly lower
            percentiles = [15, 40, 65]
            print(f"   [Adaptive] Using medium-grained percentiles: {percentiles}")
        else:
            # Fine-grained (CIFAR-100): Much lower percentiles
            percentiles = [10, 30, 60]
            print(f"   [Adaptive] Using fine-grained percentiles: {percentiles}")
        
        thresholds = [float(np.percentile(distances, p)) for p in percentiles]
        
        # Safety: ensure minimum thresholds
        thresholds = [max(t, 0.5) for t in thresholds]
        
        return thresholds
    
    def _class_aware_clustering(self, features, uncertainties, densities, predictions, thresholds):
        """
        Class-aware multi-scale clustering
        
        Ensures all classes get some representation
        """
        all_leaders = set()
        unique_classes = np.unique(predictions)
        
        # For fine-grained problems, do stratified selection
        if self.num_classes > 50:
            print(f"   [Stratified] Running class-aware selection for {self.num_classes} classes...")
            # Allocate leaders per class proportionally
            leaders_per_class = max(1, self.budget // (len(unique_classes) * 2))
            
            for class_id in unique_classes:
                class_mask = predictions == class_id
                class_indices = np.where(class_mask)[0]
                
                if len(class_indices) == 0:
                    continue
                
                # Run mini clustering within this class
                class_leaders = self._mini_clustering(
                    class_indices, features, uncertainties, densities, 
                    thresholds, min(leaders_per_class, len(class_indices))
                )
                all_leaders.update(class_leaders)
            
            return list(all_leaders)
        else:
            # Coarse-grained: use original multi-scale approach
            return self._original_multi_scale_clustering(
                features, uncertainties, densities, thresholds
            )
    
    def _mini_clustering(self, indices, features, uncertainties, densities, thresholds, budget):
        """Run clustering on a subset of points (within one class)"""
        leaders = []
        leader_features = []
        
        # Use medium threshold for within-class clustering
        threshold = thresholds[1]  # 50th percentile
        
        # Sort by uncertainty (descending)
        indices_sorted = sorted(indices, key=lambda i: uncertainties[i], reverse=True)
        
        for idx in indices_sorted:
            if len(leaders) >= budget:
                break
                
            feature = features[idx]
            
            if len(leader_features) == 0:
                leaders.append(idx)
                leader_features.append(feature)
            else:
                # Check distance to existing leaders
                distances = [np.linalg.norm(feature - lf) for lf in leader_features]
                if min(distances) > threshold:
                    leaders.append(idx)
                    leader_features.append(feature)
        
        return leaders
    
    def _original_multi_scale_clustering(self, features, uncertainties, densities, thresholds):
        """Original multi-scale clustering for coarse-grained problems"""
        all_leaders = set()
        max_leaders_per_scale = max(1000, self.budget * 10)
        
        for threshold in thresholds:
            leaders = []
            leader_features = []
            
            for i in range(len(features)):
                if len(leaders) >= max_leaders_per_scale:
                    break
                    
                feature = features[i]
                
                if len(leader_features) == 0:
                    is_leader = True
                else:
                    distances = [np.linalg.norm(feature - lf) for lf in leader_features]
                    is_leader = min(distances) > threshold
                
                if is_leader:
                    # Score based on uncertainty and moderate density
                    unc = uncertainties[i]
                    dens = densities[i]
                    
                    # Prefer high uncertainty, moderate density
                    density_score = 1.0 / (1.0 + abs(dens - np.median(densities)))
                    score = unc * 0.7 + density_score * 0.3
                    
                    if score > 0.5:  # Threshold for quality
                        leaders.append(i)
                        leader_features.append(feature)
            
            all_leaders.update(leaders)
        
        return list(all_leaders)
    
    def _score_and_select_diverse(self, candidates, features, uncertainties, densities, predictions):
        """Score candidates with class diversity consideration"""
        # Score each candidate
        scored = []
        for idx in candidates:
            unc_score = uncertainties[idx]
            dens = densities[idx]
            density_score = 1.0 / (1.0 + abs(dens - np.median(densities)))
            
            # Combined score
            score = unc_score * 0.6 + density_score * 0.4
            scored.append((idx, score, predictions[idx]))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # For fine-grained, ensure class diversity
        if self.num_classes > 50:
            selected = []
            class_counts = {}
            max_per_class = max(1, self.budget // self.num_classes + 1)
            
            for idx, score, pred_class in scored:
                if len(selected) >= self.budget:
                    break
                    
                count = class_counts.get(pred_class, 0)
                if count < max_per_class:
                    selected.append(idx)
                    class_counts[pred_class] = count + 1
            
            # If still need more, add highest scoring regardless of class
            if len(selected) < self.budget:
                for idx, score, pred_class in scored:
                    if idx not in selected:
                        selected.append(idx)
                        if len(selected) >= self.budget:
                            break
            
            return selected
        else:
            # Coarse-grained: just take top scored
            return [idx for idx, score, _ in scored[:self.budget]]
    
    def _fill_with_stratified_uncertainty(self, selected, predictions, uncertainties, num_samples):
        """Fill remaining budget with stratified uncertainty sampling"""
        remaining = [i for i in range(num_samples) if i not in selected]
        needed = self.budget - len(selected)
        
        if needed <= 0:
            return selected
        
        # Group by predicted class
        class_groups = {}
        for idx in remaining:
            pred_class = predictions[idx]
            if pred_class not in class_groups:
                class_groups[pred_class] = []
            class_groups[pred_class].append((idx, uncertainties[idx]))
        
        # Sort each class by uncertainty
        for pred_class in class_groups:
            class_groups[pred_class].sort(key=lambda x: x[1], reverse=True)
        
        # Round-robin selection across classes
        added = []
        class_ids = list(class_groups.keys())
        idx_per_class = {c: 0 for c in class_ids}
        
        while len(added) < needed and len(class_ids) > 0:
            for class_id in class_ids[:]:
                if len(added) >= needed:
                    break
                    
                idx_in_class = idx_per_class[class_id]
                if idx_in_class < len(class_groups[class_id]):
                    sample_idx, _ = class_groups[class_id][idx_in_class]
                    added.append(sample_idx)
                    idx_per_class[class_id] += 1
                else:
                    class_ids.remove(class_id)
        
        return selected + added
