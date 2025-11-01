"""
Active Learning Strategies for ICLR 2018 Core-Set Paper Reproduction

Target Results (from your specification):
- Random: baseline
- Greedy K-Center: 87.84% accuracy, 5549.29s per round (O(N²) due to distance matrix)
- Basic Leader: 86.90% accuracy, 14.31s per round (O(N))
- Advanced Leader: 89.89% accuracy, 21.48s per round (CIFAR-10), but FAILS on CIFAR-100
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import random


class RandomSampling:
    """
    Random Sampling - Baseline
    Simply selects random samples from unlabeled pool
    """
    def __init__(self, N, budget):
        self.N = N
        self.budget = budget
    
    def select_batch(self, model, unlabeled_data):
        """Randomly select budget samples"""
        num_samples = len(unlabeled_data)
        selected = np.random.choice(num_samples, min(self.budget, num_samples), replace=False)
        return list(selected)


class GreedyKCenter:
    """
    Greedy K-Center - Paper's main baseline
    
    **Algorithm:**
    1. Compute FULL N×N distance matrix (this is the bottleneck!)
    2. Start with random point
    3. Repeat budget times:
       - Select point farthest from all selected points
       - Update minimum distances
    
    **Time Complexity:** O(N²×D) for distance matrix + O(budget×N) for selection
    - The O(N²) distance matrix is why this is so slow (5549s per round)
    - This is the EXACT paper implementation - no optimizations
    """
    def __init__(self, N, budget):
        self.N = N
        self.budget = budget
    
    def select_batch(self, model, unlabeled_data):
        model.eval()
        device = next(model.parameters()).device
        
        # Extract features
        print("   Extracting features for Greedy K-Center...")
        features = self._extract_features(model, unlabeled_data, device)
        num_samples = len(features)
        
        # PAPER'S APPROACH: Compute FULL pairwise distance matrix O(N²)
        print(f"   Computing {num_samples}×{num_samples} distance matrix (this is slow!)...")
        distance_matrix = self._compute_distance_matrix(features)
        print(f"   Distance matrix size: {distance_matrix.nbytes / 1e6:.1f} MB")
        
        # Greedy K-Center selection
        selected = []
        
        # Start with random point
        first_idx = np.random.choice(num_samples)
        selected.append(first_idx)
        
        # Initialize minimum distances using precomputed matrix
        min_distances = distance_matrix[first_idx].copy()
        
        # Greedy selection
        print(f"   Greedy K-Center selection ({self.budget} samples)...")
        for _ in tqdm(range(self.budget - 1), desc="   Selecting", leave=False):
            # Find farthest point from all selected
            farthest_idx = np.argmax(min_distances)
            selected.append(farthest_idx)
            
            # Update distances using precomputed matrix
            min_distances = np.minimum(min_distances, distance_matrix[farthest_idx])
        
        return selected
    
    def _extract_features(self, model, data, device):
        """Extract feature representations"""
        features = []
        loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc="   Features", leave=False):
                inputs = inputs.to(device)
                _, feat = model(inputs)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _compute_distance_matrix(self, features):
        """
        Compute full N×N distance matrix
        This is the O(N²) bottleneck that makes Greedy K-Center slow
        """
        # Efficient computation: ||a-b||² = ||a||² + ||b||² - 2(a·b)
        norms_squared = np.sum(features ** 2, axis=1, keepdims=True)
        distance_matrix = np.sqrt(
            np.maximum(
                norms_squared + norms_squared.T - 2 * np.dot(features, features.T),
                0
            )
        )
        return distance_matrix


class LeaderClustering:
    """
    Basic Leader Clustering - Fast and Simple
    
    **Algorithm (simplified explanation):**
    1. Compute threshold from sample pairwise distances (70th percentile)
    2. Start: First unlabeled point becomes first leader
    3. For each remaining point:
       - If distance > threshold from ALL leaders → make it a new leader
       - Else → assign to closest leader's cluster
    4. If leaders < budget: fill from largest clusters
    
    **Time Complexity:** O(N×L×D) where L = number of leaders ≤ budget
    - Typically O(N) because L is small compared to N
    - Much faster than Greedy K-Center's O(N²)
    - Why fast: Only computes distances to leaders, not all pairs
    
    **Key Idea:** Instead of computing full distance matrix like Greedy,
    we only track distances to "leaders" (representative points)
    """
    def __init__(self, N, budget, threshold_percentile=70):
        self.N = N
        self.budget = budget
        self.threshold_percentile = threshold_percentile
    
    def select_batch(self, model, unlabeled_data):
        model.eval()
        device = next(model.parameters()).device
        
        # Extract features
        print("   Extracting features for Leader Clustering...")
        features = self._extract_features(model, unlabeled_data, device)
        num_samples = len(features)
        
        # Compute adaptive threshold
        threshold = self._compute_threshold(features)
        print(f"   Threshold (p{self.threshold_percentile}): {threshold:.4f}")
        
        # Leader clustering
        leaders = []
        leader_features = []
        clusters = [[] for _ in range(self.budget)]  # Pre-allocate clusters
        
        # IMPORTANT: First point becomes first leader (as you specified)
        leaders.append(0)
        leader_features.append(features[0])
        clusters[0].append(0)
        
        # Process remaining points
        print(f"   Leader clustering ({num_samples} points)...")
        for i in tqdm(range(1, num_samples), desc="   Clustering", leave=False):
            feature = features[i]
            
            # Compute distances to all current leaders
            distances = [np.linalg.norm(feature - lf) for lf in leader_features]
            min_dist = min(distances)
            closest_leader = np.argmin(distances)
            
            # Check if this should be a new leader
            if min_dist > threshold and len(leaders) < self.budget:
                # New leader: distance > threshold
                leaders.append(i)
                leader_features.append(feature)
                clusters[len(leaders)-1].append(i)
            else:
                # Assign to closest leader
                clusters[closest_leader].append(i)
        
        print(f"   Found {len(leaders)} leaders")
        
        # Fill budget if needed
        selected = leaders.copy()
        
        if len(selected) < self.budget:
            # Sort clusters by size (largest first)
            cluster_sizes = [(len(clusters[i]), i) for i in range(len(leaders))]
            cluster_sizes.sort(reverse=True)
            
            for size, cluster_idx in cluster_sizes:
                if len(selected) >= self.budget:
                    break
                
                # Add non-leader points from this cluster
                non_leaders = [p for p in clusters[cluster_idx] if p not in selected]
                needed = self.budget - len(selected)
                selected.extend(non_leaders[:needed])
        
        return selected[:self.budget]
    
    def _extract_features(self, model, data, device):
        """Extract features"""
        features = []
        loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc="   Features", leave=False):
                inputs = inputs.to(device)
                _, feat = model(inputs)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _compute_threshold(self, features):
        """
        Compute adaptive threshold as percentile of pairwise distances
        This determines when a point is "far enough" to be a new leader
        """
        sample_size = min(500, len(features))
        sample_idx = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features[sample_idx]
        
        # Sample pairwise distances
        distances = []
        for i in range(min(100, len(sample_features))):
            dists = np.linalg.norm(sample_features[i] - sample_features, axis=1)
            distances.extend(dists[dists > 0])  # Exclude self-distance
        
        return np.percentile(distances, self.threshold_percentile)


class AdvancedLeader:
    """
    VERSION 2: Advanced Leader Clustering with Volatility Reduction
    
    **Design Philosophy:**
    - UNIVERSAL: No dataset-specific conditional logic
    - DATA-DRIVEN: All adaptations based on measured characteristics
    - STABLE: Smooth transitions between rounds, controlled randomness
    - BALANCED: Maintain diversity (leaders) + uncertainty trade-off
    
    **Key Improvements in V2:**
    1. Smooth CV-based threshold adaptation (no discrete jumps)
    2. Minimum leader target (50% of budget)
    3. Controlled 70/30 leader/uncertainty balance
    4. Threshold momentum for temporal smoothing (30% previous)
    
    **Algorithm Flow:**
    1. Extract features + uncertainty + predictions
    2. Compute adaptive thresholds with smoothing
    3. Ensure minimum 50% of budget comes from leaders
    4. Select 70% from diversity-based leaders
    5. Fill 30% with stratified uncertainty sampling
    
    **Expected Results:**
    - Reduced volatility (stable accuracy across rounds)
    - Maintained final performance (≥39% CIFAR-100)
    - Works universally on any dataset
    """
    def __init__(self, N, budget):
        self.N = N
        self.budget = budget
        self.num_classes = None  # Auto-detected from predictions
        self.prev_thresholds = None  # For temporal smoothing
        self.momentum = 0.3  # 30% weight to previous thresholds
    
    def select_batch(self, model, unlabeled_data):
        model.eval()
        device = next(model.parameters()).device
        
        # Extract features, uncertainties, AND predictions
        print("   [V2] Extracting features + uncertainties + predictions...")
        features, uncertainties, predictions = self._extract_features_and_uncertainty(
            model, unlabeled_data, device
        )
        num_samples = len(features)
        
        # Detect number of classes from predictions
        self.num_classes = len(np.unique(predictions))
        print(f"   [V2] Detected {self.num_classes} classes in unlabeled data")
        
        # Compute local densities (adaptive k)
        print("   Computing local densities (adaptive k-NN)...")
        densities = self._compute_densities(features)
        
        # Compute ADAPTIVE multi-scale thresholds with MOMENTUM SMOOTHING
        thresholds = self._compute_multi_scale_thresholds_v2(features)
        print(f"   Multi-scale thresholds: {[f'{t:.3f}' for t in thresholds]}")
        
        # TARGET: Get enough leaders (minimum 50% of budget)
        target_leader_budget = max(int(self.budget * 0.7), 1)  # Target 70% from leaders
        min_leader_budget = max(int(self.budget * 0.5), 1)     # Minimum 50% from leaders
        
        print(f"   [V2] Target leaders: {target_leader_budget} (min {min_leader_budget})")
        
        # Multi-scale clustering with adaptive threshold relaxation
        candidate_leaders = self._multi_scale_clustering_v2(
            features, uncertainties, densities, thresholds, predictions,
            target_budget=target_leader_budget,
            min_budget=min_leader_budget
        )
        
        print(f"   Candidate leaders: {len(candidate_leaders)}")
        
        # Select up to target_leader_budget from candidates
        if len(candidate_leaders) > target_leader_budget:
            selected_leaders = self._score_and_select(
                candidate_leaders, features, uncertainties, densities, predictions,
                budget=target_leader_budget
            )
        else:
            selected_leaders = candidate_leaders
        
        print(f"   [V2] Selected {len(selected_leaders)} leaders ({len(selected_leaders)/self.budget*100:.1f}% of budget)")
        
        # Fill remaining with STRATIFIED uncertainty (30% target)
        uncertainty_budget = self.budget - len(selected_leaders)
        if uncertainty_budget > 0:
            final_selection = self._fill_with_stratified_uncertainty(
                selected_leaders, predictions, uncertainties, num_samples,
                fill_budget=uncertainty_budget
            )
        else:
            final_selection = selected_leaders[:self.budget]
        
        print(f"   Final selection: {len(final_selection)} samples "
              f"({len(selected_leaders)} leaders + {len(final_selection)-len(selected_leaders)} uncertainty)")
        
        return final_selection[:self.budget]
    
    def _extract_features_and_uncertainty(self, model, data, device):
        """Extract features, uncertainties, AND predictions for class-aware sampling"""
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
                
                # Predictions: for class-aware sampling
                preds = torch.argmax(outputs, dim=1)
                predictions.append(preds.cpu().numpy())
        
        return (np.concatenate(features, axis=0), 
                np.concatenate(uncertainties, axis=0),
                np.concatenate(predictions, axis=0))
    
    def _compute_densities(self, features):
        """
        UNIVERSAL Adaptive Density Estimation
        
        Key Insight: Don't use fixed k=10!
        Instead, adaptively choose k based on dataset size and local structure.
        
        For small unlabeled pools: use smaller k
        For large unlabeled pools: use larger k
        
        This naturally works for any dataset without knowing num_classes!
        """
        N = len(features)
        
        if N < 10:
            return np.ones(N)
        
        # ADAPTIVE k based on sample size, not num_classes
        # Use roughly sqrt(N) neighbors, but cap between 10 and 50
        adaptive_k = max(10, min(50, int(np.sqrt(N))))
        
        # Further refine: if data seems sparse (many dimensions, few points)
        # use smaller k to avoid spanning too much
        dim = features.shape[1]
        if N < dim:  # More dimensions than points - very sparse!
            adaptive_k = max(5, min(adaptive_k, N // 2))
        
        print(f"   Adaptive k={adaptive_k} (N={N}, dim={dim})")
        
        nbrs = NearestNeighbors(n_neighbors=min(adaptive_k, N-1)).fit(features)
        distances, _ = nbrs.kneighbors(features)
        
        avg_dist = np.mean(distances, axis=1)
        densities = 1.0 / (avg_dist + 1e-8)
        return densities
    
    def _compute_multi_scale_thresholds_v2(self, features):
        """
        VERSION 2: Smooth Adaptive Thresholds with Temporal Momentum
        
        Key Improvements:
        1. SMOOTH interpolation (no discrete CV buckets)
        2. CONSERVATIVE percentiles (higher → fewer leaders → better quality)
        3. TEMPORAL smoothing (momentum with previous round)
        
        This reduces volatility while maintaining adaptiveness!
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
                if i == j:
                    continue
                distances.append(np.linalg.norm(sample_features[i] - sample_features[j]))

        if len(distances) == 0:
            return [0.5, 1.0, 1.5]

        distances = np.array(distances)
        
        # Measure distribution spread
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        cv = std_dist / (mean_dist + 1e-8)
        
        # SMOOTH CONTINUOUS INTERPOLATION (key improvement!)
        # Map CV [0.2, 0.8] → percentiles
        # Low CV (0.2) = overlapping → use lower percentiles [20, 40, 65]
        # High CV (0.8) = separated → use higher percentiles [30, 55, 75]
        
        cv_clamped = np.clip(cv, 0.2, 0.8)
        t = (cv_clamped - 0.2) / 0.6  # Normalize to [0, 1]
        
        # Smooth interpolation (MORE CONSERVATIVE than V1)
        # V1 used [15, 35, 60] for low CV - TOO AGGRESSIVE
        # V2 uses [20, 40, 65] for low CV - MORE SELECTIVE
        fine_p = 20 + t * 10      # 20 → 30
        medium_p = 40 + t * 15    # 40 → 55  
        coarse_p = 65 + t * 10    # 65 → 75
        
        # Compute thresholds
        p_fine = float(np.percentile(distances, fine_p))
        p_med = float(np.percentile(distances, medium_p))
        p_coarse = float(np.percentile(distances, coarse_p))
        
        new_thresholds = [p_fine, p_med, p_coarse]
        
        # TEMPORAL SMOOTHING (key improvement!)
        # Blend with previous round using exponential moving average
        if self.prev_thresholds is not None:
            smoothed_thresholds = [
                self.momentum * prev + (1 - self.momentum) * new
                for prev, new in zip(self.prev_thresholds, new_thresholds)
            ]
        else:
            smoothed_thresholds = new_thresholds
        
        # Store for next round
        self.prev_thresholds = smoothed_thresholds
        
        # Safety: ensure minimum thresholds
        smoothed_thresholds = [
            max(smoothed_thresholds[0], 0.5),
            max(smoothed_thresholds[1], 1.0),
            max(smoothed_thresholds[2], 2.0)
        ]
        
        print(f"   CV={cv:.3f} → Percentiles=[{fine_p:.0f}, {medium_p:.0f}, {coarse_p:.0f}]")
        if self.prev_thresholds:
            print(f"   Raw: [{p_fine:.3f}, {p_med:.3f}, {p_coarse:.3f}] "
                  f"→ Smoothed: {[f'{t:.3f}' for t in smoothed_thresholds]}")
        
        return smoothed_thresholds
    
    def _multi_scale_clustering_v2(self, features, uncertainties, densities, thresholds, 
                                    predictions, target_budget, min_budget):
        """
        VERSION 2: Multi-scale Clustering with Minimum Leader Target
        
        Key Improvement: Iteratively RELAX thresholds if we don't get enough leaders
        This ensures at least min_budget (50%) comes from diversity-based selection
        """
        max_attempts = 5
        current_thresholds = thresholds.copy()
        
        for attempt in range(max_attempts):
            # Try clustering with current thresholds
            leaders = self._multi_scale_clustering(
                features, uncertainties, densities, current_thresholds, predictions
            )
            
            # Check if we have enough leaders
            if len(leaders) >= min_budget:
                if attempt > 0:
                    print(f"   [V2] Relaxed thresholds {attempt} times to get {len(leaders)} leaders")
                return leaders
            
            # Not enough leaders - relax thresholds by 25%
            current_thresholds = [t * 1.25 for t in current_thresholds]
            print(f"   [V2] Attempt {attempt+1}: Only {len(leaders)} leaders, "
                  f"relaxing thresholds to {[f'{t:.2f}' for t in current_thresholds]}")
        
        # After max attempts, return what we have
        print(f"   [V2] WARNING: After {max_attempts} attempts, only got {len(leaders)} leaders (min was {min_budget})")
        return leaders
    
    def _multi_scale_clustering(self, features, uncertainties, densities, thresholds, predictions):
        """
        Base multi-scale clustering implementation
        
        Decides between stratified (many classes) or standard (few classes) approach
        """
        # For fine-grained (many classes), use stratified approach
        if self.num_classes > 50:
            return self._stratified_clustering(
                features, uncertainties, densities, thresholds[0], predictions
            )
        
        # For coarse-grained (few classes), use original multi-scale
        all_leaders = set()
        scale_weights = [1.0, 0.7, 0.4]  # Prioritize fine scale
        
        # Cap leaders per scale to prevent explosion
        max_leaders_per_scale = max(1000, self.budget * 10)
        
        for scale_idx, threshold in enumerate(thresholds):
            leaders = []
            leader_features = []
            weight = scale_weights[scale_idx] if scale_idx < len(scale_weights) else 0.2
            
            for i in range(len(features)):
                feature = features[i]
                
                # Check distance to leaders at this scale
                if len(leader_features) == 0:
                    is_leader = True
                else:
                    distances = [np.linalg.norm(feature - lf) for lf in leader_features]
                    is_leader = min(distances) > threshold
                
                if is_leader:
                    # Score: high uncertainty + moderate density
                    unc_score = uncertainties[i]
                    dens_score = 1.0 - densities[i] / (np.max(densities) + 1e-8)
                    combined_score = unc_score * 0.6 + dens_score * 0.4
                    
                    leaders.append((i, combined_score * weight))
                    leader_features.append(feature)

                # Safety cap
                if len(leader_features) >= max_leaders_per_scale:
                    break
            
            all_leaders.update([idx for idx, _ in leaders])
        
        return list(all_leaders)
    
    def _stratified_clustering(self, features, uncertainties, densities, threshold, predictions):
        """
        Stratified clustering for fine-grained problems
        
        Ensures each predicted class gets leaders proportionally
        Prevents dense classes from dominating
        """
        unique_classes = np.unique(predictions)
        num_classes = len(unique_classes)
        
        # Target leaders per class (with minimum)
        target_per_class = max(3, self.budget // (num_classes * 3))
        max_per_class = self.budget // num_classes + 10
        
        print(f"   [Stratified] Target {target_per_class}-{max_per_class} leaders per class")
        
        all_leaders = []
        class_counts = []
        
        for class_id in unique_classes:
            class_mask = predictions == class_id
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Run clustering within this class
            class_features = features[class_indices]
            class_uncertainties = uncertainties[class_indices]
            class_densities = densities[class_indices]
            
            # Find leaders within this class
            leaders = []
            leader_features = []
            
            for i, global_idx in enumerate(class_indices):
                feature = class_features[i]
                
                if len(leader_features) == 0:
                    is_leader = True
                else:
                    distances = [np.linalg.norm(feature - lf) for lf in leader_features]
                    is_leader = min(distances) > threshold
                
                if is_leader:
                    unc_score = class_uncertainties[i]
                    dens_score = 1.0 - class_densities[i] / (np.max(class_densities) + 1e-8)
                    score = unc_score * 0.7 + dens_score * 0.3
                    
                    leaders.append((global_idx, score))
                    leader_features.append(feature)
                    
                    # Cap per class
                    if len(leaders) >= max_per_class:
                        break
            
            # Sort by score and take top targets
            leaders.sort(key=lambda x: x[1], reverse=True)
            selected_from_class = [idx for idx, _ in leaders[:target_per_class]]
            all_leaders.extend(selected_from_class)
            class_counts.append(len(selected_from_class))
        
        print(f"   [Stratified] Selected from {len(class_counts)} classes, "
              f"avg {np.mean(class_counts):.1f} per class")
        
        return all_leaders
    
    def _score_and_select(self, candidates, features, uncertainties, densities, predictions, budget=None):
        """
        Score candidates by:
        - Uncertainty (35%): prefer uncertain samples
        - Density (25%): prefer low density regions
        - Diversity (25%): prefer distant from already selected
        - Class balance (15%): prefer under-represented classes (for fine-grained)
        """
        if budget is None:
            budget = self.budget
            
        # Track class counts for balancing
        class_counts = {}
        if self.num_classes > 50:
            # For fine-grained, track class representation
            for pred in predictions[candidates]:
                class_counts[pred] = class_counts.get(pred, 0) + 1
        
        scores = []
        selected = []
        selected_classes = []
        
        for idx in candidates:
            # Uncertainty
            unc_score = uncertainties[idx]
            
            # Density (prefer lower)
            dens_score = 1.0 - densities[idx] / (np.max(densities) + 1e-8)
            
            # Diversity
            if len(selected) == 0:
                div_score = 1.0
            else:
                selected_features = features[selected]
                distances = [np.linalg.norm(features[idx] - sf) for sf in selected_features]
                div_score = np.mean(distances) / (np.max(distances) + 1e-8)
            
            # Class balance (for fine-grained only)
            if self.num_classes > 50:
                pred_class = predictions[idx]
                class_freq = selected_classes.count(pred_class) if selected_classes else 0
                # Penalize over-represented classes
                balance_score = 1.0 / (1.0 + class_freq)
            else:
                balance_score = 1.0
            
            # Combined score
            total_score = (unc_score * 0.35 + dens_score * 0.25 + 
                          div_score * 0.25 + balance_score * 0.15)
            scores.append((idx, total_score))
        
        # Select top candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in scores[:budget]]
        return selected
    
    def _fill_with_stratified_uncertainty(self, selected, predictions, uncertainties, num_samples, fill_budget=None):
        """
        Fill remaining budget with stratified uncertainty sampling
        
        For fine-grained problems, ensures all classes get representation
        For coarse-grained, uses simple uncertainty sampling
        """
        if fill_budget is None:
            fill_budget = self.budget - len(selected)
            
        remaining = fill_budget
        if remaining <= 0:
            return selected
        
        print(f"   [Fill] Need {remaining} more samples")
        
        # Get candidates not already selected
        all_indices = set(range(num_samples))
        selected_set = set(selected)
        candidates = list(all_indices - selected_set)
        
        if self.num_classes > 50:
            # Stratified: ensure each class gets samples
            unique_classes = np.unique(predictions)
            samples_per_class = max(1, remaining // len(unique_classes))
            
            additional = []
            for class_id in unique_classes:
                class_mask = predictions == class_id
                class_candidates = [idx for idx in candidates if class_mask[idx]]
                
                if len(class_candidates) == 0:
                    continue
                
                # Sort by uncertainty
                class_unc = [(idx, uncertainties[idx]) for idx in class_candidates]
                class_unc.sort(key=lambda x: x[1], reverse=True)
                
                # Take top uncertain from this class
                class_selected = [idx for idx, _ in class_unc[:samples_per_class]]
                additional.extend(class_selected)
                
                if len(additional) >= remaining:
                    break
            
            print(f"   [Fill] Stratified: added {len(additional[:remaining])} samples")
            return selected + additional[:remaining]
        else:
            # Simple uncertainty sampling
            candidate_unc = [(idx, uncertainties[idx]) for idx in candidates]
            candidate_unc.sort(key=lambda x: x[1], reverse=True)
            additional = [idx for idx, _ in candidate_unc[:remaining]]
            
            print(f"   [Fill] Uncertainty: added {len(additional)} samples")
            return selected + additional
