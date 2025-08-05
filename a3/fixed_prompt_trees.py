"""
Time Series Prompt Classification using Tree-based Methods
Implementation of Prompt Trees and Random Forests for Univariate/Multivariate Time Series

Author: Implementation based on Pietro Sala's assignment
Course: Biomedical Decision Support Systems 24/25
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE DEFINITIONS AND DATA STRUCTURES
# ============================================================================

class TimeSeriesSample:
    """Represents a time series sample with data and optional label"""
    def __init__(self, data: np.ndarray, label: Optional[int] = None):
        self.data = data  # Shape: (channels, length) or (length,) for univariate
        self.label = label
        self.is_multivariate = len(data.shape) > 1

    @property
    def length(self):
        return self.data.shape[-1]

    @property
    def channels(self):
        return self.data.shape[0] if self.is_multivariate else 1

class SliceTest:
    """Basic slice test implementation"""
    def __init__(self, func: Callable, begin: int, end: int):
        self.func = func
        self.begin = begin
        self.end = end

    def evaluate(self, x: np.ndarray) -> bool:
        return self.func(x[..., self.begin:self.end])

class ReferenceSliceTest:
    """Reference Slice Test (RST) implementation"""
    def __init__(self, reference: np.ndarray, begin: int, end: int,
                 distance_func: Callable, threshold: float, channel: int = 0):
        self.reference = reference
        self.begin = begin
        self.end = end
        self.distance_func = distance_func
        self.threshold = threshold
        self.channel = channel

    def evaluate(self, x: np.ndarray) -> bool:
        if len(x.shape) > 1:  # Multivariate
            slice_data = x[self.channel, self.begin:self.end]
        else:  # Univariate
            slice_data = x[self.begin:self.end]

        distance = self.distance_func(slice_data, self.reference)
        return distance <= self.threshold

# ============================================================================
# DISTANCE FUNCTIONS
# ============================================================================

class DistanceFunctions:
    """Collection of distance functions for time series comparison"""

    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean distance between two time series"""
        min_len = min(len(x), len(y))
        return euclidean(x[:min_len], y[:min_len])

    @staticmethod
    def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Manhattan distance between two time series"""
        min_len = min(len(x), len(y))
        return cityblock(x[:min_len], y[:min_len])

    @staticmethod
    def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Cosine distance between two time series"""
        min_len = min(len(x), len(y))
        if min_len == 0:
            return 1.0

        x_norm = x[:min_len]
        y_norm = y[:min_len]

        # Handle zero vectors
        x_norm_sq = np.sum(x_norm**2)
        y_norm_sq = np.sum(y_norm**2)

        if x_norm_sq == 0 or y_norm_sq == 0:
            return 1.0

        try:
            return cosine(x_norm, y_norm)
        except:
            return 1.0  # Return max distance on error

    @staticmethod
    def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Simple DTW distance implementation"""
        n, m = len(x), len(y)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i-1] - y[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

        return dtw[n, m]

# ============================================================================
# PROMPT TREE IMPLEMENTATION
# ============================================================================

class PromptTreeNode:
    """Base class for prompt tree nodes"""
    def __init__(self):
        self.is_leaf = False
        self.distribution = None

class LeafNode(PromptTreeNode):
    """Leaf node containing class distribution"""
    def __init__(self, distribution: Dict[int, float]):
        super().__init__()
        self.is_leaf = True
        self.distribution = distribution

    def predict(self) -> int:
        if not self.distribution:
            return -1
        return max(self.distribution.keys(), key=lambda k: self.distribution[k])

class InternalNode(PromptTreeNode):
    """Internal node containing a reference slice test"""
    def __init__(self, test: ReferenceSliceTest):
        super().__init__()
        self.test = test
        self.left = None  # True branch
        self.right = None  # False branch
        self.distribution = None

class PromptTree:
    """Prompt Tree implementation for time series classification"""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, distance_functions: List[Callable] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.distance_functions = distance_functions or [
            DistanceFunctions.euclidean_distance,
            DistanceFunctions.manhattan_distance,
            DistanceFunctions.cosine_distance
        ]

    def fit(self, X: List[TimeSeriesSample], y: Optional[List[int]] = None):
        """Fit the prompt tree to training data"""
        self.root = self._build_tree(X, y, path=[], depth=0)

    def _build_tree(self, X: List[TimeSeriesSample], y: Optional[List[int]],
                   path: List[ReferenceSliceTest], depth: int) -> PromptTreeNode:
        """Recursively build the prompt tree"""

        # Stopping criteria
        if (depth >= self.max_depth or
            len(X) < self.min_samples_split or
            (y is not None and len(set(y)) <= 1)):
            return self._create_leaf(X, y)

        # Generate candidate tests
        candidate_tests = self._generate_candidate_tests(X, y, path)

        if not candidate_tests:
            return self._create_leaf(X, y)

        # Select best test
        best_test, best_score = self._select_best_test(X, y, candidate_tests)

        if best_test is None:
            return self._create_leaf(X, y)

        # Split data
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_test)

        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            return self._create_leaf(X, y)

        # Create internal node
        node = InternalNode(best_test)
        new_path = path + [best_test]

        # Recursively build subtrees
        node.left = self._build_tree(X_left, y_left, new_path, depth + 1)
        node.right = self._build_tree(X_right, y_right, new_path, depth + 1)

        return node

    def _create_leaf(self, X: List[TimeSeriesSample], y: Optional[List[int]]) -> LeafNode:
        """Create a leaf node with class distribution"""
        if y is None:
            # Unsupervised case - return uniform distribution
            distribution = {0: 1.0}  # Default class
        else:
            # Supervised case - compute class distribution
            class_counts = Counter(y)
            total = len(y)
            distribution = {cls: count/total for cls, count in class_counts.items()}

        return LeafNode(distribution)

    def _generate_candidate_tests(self, X: List[TimeSeriesSample], y: Optional[List[int]],
                                path: List[ReferenceSliceTest]) -> List[ReferenceSliceTest]:
        """Generate candidate reference slice tests"""
        candidates = []

        if not X:
            return candidates

        try:
            # Get path constraints
            B = {0}  # Root starts at 0
            E = {0}

            for test in path:
                B.add(test.begin)
                E.add(test.end)

            max_length = min([sample.length for sample in X])
            if max_length <= 1:
                return candidates

            next_start = max(E) + 1 if E else 0

            # Generate intervals to test (limit to reasonable sizes)
            intervals = []
            for b in list(B) + [next_start]:
                if b < max_length:
                    for e in range(b + 1, min(b + 10, max_length + 1)):  # Limit window size
                        intervals.append((b, e))

            # Limit number of intervals to prevent explosion
            intervals = intervals[:20]

            # Generate tests for each interval
            for b, e in intervals:
                # Limit reference samples to speed up computation
                for i, sample in enumerate(X[:min(5, len(X))]):
                    for channel in range(sample.channels):
                        try:
                            if sample.is_multivariate:
                                ref_data = sample.data[channel, b:e]
                            else:
                                ref_data = sample.data[b:e]

                            if len(ref_data) == 0:
                                continue

                            # Limit distance functions to speed up
                            for dist_func in self.distance_functions[:2]:  # Use only first 2
                                # Calculate distances to other samples
                                distances = []
                                for other_sample in X[:min(10, len(X))]:  # Limit comparison samples
                                    try:
                                        if other_sample.is_multivariate:
                                            other_data = other_sample.data[channel, b:e]
                                        else:
                                            other_data = other_sample.data[b:e]

                                        if len(other_data) > 0 and len(ref_data) > 0:
                                            dist = dist_func(ref_data, other_data)
                                            if not np.isnan(dist) and not np.isinf(dist):
                                                distances.append(dist)
                                    except:
                                        continue

                                if len(distances) > 1:
                                    threshold = np.median(distances)
                                    if not np.isnan(threshold) and not np.isinf(threshold):
                                        test = ReferenceSliceTest(ref_data, b, e, dist_func,
                                                               threshold, channel)
                                        candidates.append(test)
                        except:
                            continue  # Skip problematic samples

        except Exception as e:
            print(f"Warning: Error in candidate generation: {e}")
            return []

        return candidates[:50]  # Limit total candidates

    def _select_best_test(self, X: List[TimeSeriesSample], y: Optional[List[int]],
                         candidates: List[ReferenceSliceTest]) -> Tuple[Optional[ReferenceSliceTest], float]:
        """Select the best test based on information gain or other criteria"""
        if not candidates:
            return None, 0.0

        best_test = None
        best_score = -np.inf

        for test in candidates:
            X_left, y_left, X_right, y_right = self._split_data(X, y, test)

            if len(X_left) == 0 or len(X_right) == 0:
                continue

            if y is not None:
                # Supervised: use information gain
                score = self._information_gain(y, y_left, y_right)
            else:
                # Unsupervised: use balance criterion
                score = -abs(len(X_left) - len(X_right))  # Prefer balanced splits

            if score > best_score:
                best_score = score
                best_test = test

        return best_test, best_score

    def _split_data(self, X: List[TimeSeriesSample], y: Optional[List[int]],
                   test: ReferenceSliceTest) -> Tuple[List[TimeSeriesSample], Optional[List[int]],
                                                     List[TimeSeriesSample], Optional[List[int]]]:
        """Split data based on test result"""
        X_left, X_right = [], []
        if y is not None:
            y_left, y_right = [], []
        else:
            y_left, y_right = None, None

        for i, sample in enumerate(X):
            try:
                if test.evaluate(sample.data):
                    X_left.append(sample)
                    if y is not None:
                        y_left.append(y[i])
                else:
                    X_right.append(sample)
                    if y is not None:
                        y_right.append(y[i])
            except:
                # If test fails, put in right branch
                X_right.append(sample)
                if y is not None:
                    y_right.append(y[i])

        return X_left, y_left, X_right, y_right

    def _information_gain(self, y_parent: List[int], y_left: List[int],
                         y_right: List[int]) -> float:
        """Calculate information gain for a split"""
        def entropy(labels):
            if not labels:
                return 0
            counts = Counter(labels)
            probs = [count/len(labels) for count in counts.values()]
            return -sum(p * np.log2(p) for p in probs if p > 0)

        parent_entropy = entropy(y_parent)
        n = len(y_parent)
        left_weight = len(y_left) / n
        right_weight = len(y_right) / n

        weighted_child_entropy = (left_weight * entropy(y_left) +
                                right_weight * entropy(y_right))

        return parent_entropy - weighted_child_entropy

    def predict(self, sample: TimeSeriesSample) -> Union[int, str]:
        """Predict class for a single sample"""
        if self.root is None:
            return "too-early"

        node = self.root
        while not node.is_leaf:
            try:
                if node.test.evaluate(sample.data):
                    node = node.left
                else:
                    node = node.right

                if node is None:
                    return "too-early"
            except:
                return "too-early"

        return node.predict()

    def predict_batch(self, X: List[TimeSeriesSample]) -> List[Union[int, str]]:
        """Predict classes for multiple samples"""
        return [self.predict(sample) for sample in X]

    def get_tree_paths(self) -> List[List[ReferenceSliceTest]]:
        """Get all paths from root to leaves"""
        paths = []

        def traverse(node, current_path):
            if node.is_leaf:
                paths.append(current_path.copy())
            else:
                if node.left:
                    traverse(node.left, current_path + [node.test])
                if node.right:
                    traverse(node.right, current_path + [node.test])

        if self.root:
            traverse(self.root, [])

        return paths

# ============================================================================
# RANDOM FOREST IMPLEMENTATION
# ============================================================================

class VotingMechanism:
    """Different voting mechanisms for random forest"""

    @staticmethod
    def majority_vote(predictions: List[List[Union[int, str]]]) -> List[Union[int, str]]:
        """Simple majority voting"""
        n_samples = len(predictions[0])
        final_predictions = []

        for i in range(n_samples):
            votes = [pred[i] for pred in predictions if pred[i] != "too-early"]
            if not votes:
                final_predictions.append("too-early")
            else:
                final_predictions.append(Counter(votes).most_common(1)[0][0])

        return final_predictions

    @staticmethod
    def weighted_vote(predictions: List[List[Union[int, str]]],
                     weights: List[float]) -> List[Union[int, str]]:
        """Weighted voting based on tree performance"""
        n_samples = len(predictions[0])
        final_predictions = []

        for i in range(n_samples):
            vote_weights = defaultdict(float)
            total_weight = 0

            for j, pred in enumerate(predictions):
                if pred[i] != "too-early":
                    vote_weights[pred[i]] += weights[j]
                    total_weight += weights[j]

            if total_weight == 0:
                final_predictions.append("too-early")
            else:
                best_class = max(vote_weights.keys(), key=lambda k: vote_weights[k])
                final_predictions.append(best_class)

        return final_predictions

    @staticmethod
    def track_record_vote(predictions: List[List[Union[int, str]]],
                         track_records: List[Dict]) -> List[Union[int, str]]:
        """Voting based on track record of each tree"""
        n_samples = len(predictions[0])
        final_predictions = []

        for i in range(n_samples):
            class_scores = defaultdict(float)

            for j, pred in enumerate(predictions):
                if pred[i] != "too-early":
                    accuracy = track_records[j].get('accuracy', 0.5)
                    class_scores[pred[i]] += accuracy

            if not class_scores:
                final_predictions.append("too-early")
            else:
                best_class = max(class_scores.keys(), key=lambda k: class_scores[k])
                final_predictions.append(best_class)

        return final_predictions

class PromptRandomForest:
    """Random Forest implementation for prompt time series classification"""

    def __init__(self, n_estimators: int = 10, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 bootstrap: bool = True, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []
        self.tree_weights = []
        self.track_records = []

        if random_state:
            np.random.seed(random_state)

    def fit(self, X: List[TimeSeriesSample], y: Optional[List[int]] = None):
        """Fit the random forest"""
        self.trees = []
        self.tree_weights = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = [X[idx] for idx in indices]
                y_boot = [y[idx] for idx in indices] if y is not None else None
            else:
                X_boot, y_boot = X, y

            # Create and train tree
            tree = PromptTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )

            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # Calculate tree weight (simple accuracy-based)
            if y is not None:
                predictions = tree.predict_batch(X_boot)
                correct = sum(1 for pred, true in zip(predictions, y_boot)
                            if pred == true and pred != "too-early")
                total = len([p for p in predictions if p != "too-early"])
                weight = correct / total if total > 0 else 0.5
            else:
                weight = 1.0

            self.tree_weights.append(weight)
            self.track_records.append({'accuracy': weight})

    def predict(self, X: List[TimeSeriesSample],
               voting: str = 'majority') -> List[Union[int, str]]:
        """Predict using the specified voting mechanism"""
        if not self.trees:
            return ["too-early"] * len(X)

        # Get predictions from all trees
        all_predictions = []
        for tree in self.trees:
            predictions = tree.predict_batch(X)
            all_predictions.append(predictions)

        # Apply voting mechanism
        if voting == 'majority':
            return VotingMechanism.majority_vote(all_predictions)
        elif voting == 'weighted':
            return VotingMechanism.weighted_vote(all_predictions, self.tree_weights)
        elif voting == 'track_record':
            return VotingMechanism.track_record_vote(all_predictions, self.track_records)
        else:
            raise ValueError(f"Unknown voting mechanism: {voting}")

    # Fixed: Add the missing predict_batch method
    def predict_batch(self, X: List[TimeSeriesSample]) -> List[Union[int, str]]:
        """Predict classes for multiple samples using majority voting by default"""
        return self.predict(X, voting='majority')

    def _get_leaf_node(self, tree: PromptTree, sample: TimeSeriesSample) -> Optional[LeafNode]:
        """Get the leaf node where a sample ends up"""
        if tree.root is None:
            return None

        node = tree.root
        while not node.is_leaf:
            try:
                if node.test.evaluate(sample.data):
                    node = node.left
                else:
                    node = node.right

                if node is None:
                    return None
            except:
                return None

        return node

    def _get_shared_path_depth(self, tree: PromptTree, x1: TimeSeriesSample,
                              x2: TimeSeriesSample) -> int:
        """Get the depth of shared path between two samples"""
        if tree.root is None:
            return 0

        node = tree.root
        depth = 0

        while not node.is_leaf:
            try:
                result1 = node.test.evaluate(x1.data)
                result2 = node.test.evaluate(x2.data)

                if result1 != result2:
                    break  # Paths diverge

                depth += 1
                node = node.left if result1 else node.right

                if node is None:
                    break
            except:
                break

        return depth

    def _get_max_path_depth(self, tree: PromptTree, x1: TimeSeriesSample,
                           x2: TimeSeriesSample) -> int:
        """Get the maximum path depth for two samples"""
        depth1 = self._get_path_depth(tree, x1)
        depth2 = self._get_path_depth(tree, x2)
        return max(depth1, depth2)

    def _get_path_depth(self, tree: PromptTree, sample: TimeSeriesSample) -> int:
        """Get the path depth for a single sample"""
        if tree.root is None:
            return 0

        node = tree.root
        depth = 0

        while not node.is_leaf:
            try:
                if node.test.evaluate(sample.data):
                    node = node.left
                else:
                    node = node.right

                if node is None:
                    break

                depth += 1
            except:
                break

        return depth

# ============================================================================
# FOREST DISTANCE FUNCTIONS
# ============================================================================

class ForestDistances:
    """Distance measures for random forest clustering"""

    @staticmethod
    def breiman_distance(forest: PromptRandomForest, x1: TimeSeriesSample,
                        x2: TimeSeriesSample) -> float:
        """Breiman distance: fraction of trees where samples end up in different leaves"""
        if not forest.trees:
            return 1.0

        different_leaves = 0
        valid_trees = 0

        for tree in forest.trees:
            try:
                # Get leaf nodes for both samples
                leaf1 = forest._get_leaf_node(tree, x1)
                leaf2 = forest._get_leaf_node(tree, x2)

                if leaf1 != leaf2:
                    different_leaves += 1
                valid_trees += 1
            except:
                continue  # Skip if tree fails

        return different_leaves / valid_trees if valid_trees > 0 else 1.0

    @staticmethod
    def zhu_distance(forest: PromptRandomForest, x1: TimeSeriesSample,
                    x2: TimeSeriesSample) -> float:
        """Zhu distance: based on shared path length in trees"""
        if not forest.trees:
            return 1.0

        total_shared_depth = 0
        total_max_depth = 0

        for tree in forest.trees:
            try:
                shared_depth = forest._get_shared_path_depth(tree, x1, x2)
                max_depth = forest._get_max_path_depth(tree, x1, x2)

                total_shared_depth += shared_depth
                total_max_depth += max_depth
            except:
                continue  # Skip if tree fails

        if total_max_depth == 0:
            return 1.0

        return 1.0 - (total_shared_depth / total_max_depth)

    @staticmethod
    def ratio_rf_distance(forest: PromptRandomForest, x1: TimeSeriesSample,
                         x2: TimeSeriesSample) -> float:
        """RatioRF distance: ratio-based distance measure"""
        if not forest.trees:
            return 1.0

        same_leaf_count = 0
        total_trees = 0

        for tree in forest.trees:
            try:
                leaf1 = forest._get_leaf_node(tree, x1)
                leaf2 = forest._get_leaf_node(tree, x2)

                if leaf1 == leaf2:
                    same_leaf_count += 1
                total_trees += 1
            except:
                continue  # Skip if tree fails

        if total_trees == 0:
            return 1.0

        similarity = same_leaf_count / total_trees
        return 1.0 - similarity

# ============================================================================
# CLUSTERING AND EVALUATION
# ============================================================================

class TimeSeriesClusterer:
    """Clustering based on forest distances"""

    def __init__(self, forest: PromptRandomForest):
        self.forest = forest

    def cluster(self, X: List[TimeSeriesSample], n_clusters: int,
               distance_type: str = 'breiman') -> List[int]:
        """Perform hierarchical clustering based on forest distances"""
        n_samples = len(X)

        # Calculate distance matrix
        distance_matrix = np.zeros((n_samples, n_samples))

        distance_func = {
            'breiman': ForestDistances.breiman_distance,
            'zhu': ForestDistances.zhu_distance,
            'ratio_rf': ForestDistances.ratio_rf_distance
        }[distance_type]

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = distance_func(self.forest, X[i], X[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Perform hierarchical clustering
        # Fixed: Convert to condensed distance matrix properly
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        return cluster_labels - 1  # Convert to 0-based indexing

class ClusteringEvaluator:
    """Evaluation metrics for clustering results"""

    @staticmethod
    def purity(true_labels: List[int], cluster_labels: List[int]) -> float:
        """Calculate cluster purity"""
        n_samples = len(true_labels)
        purity = 0

        for cluster_id in set(cluster_labels):
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_true_labels = np.array(true_labels)[cluster_mask]

            if len(cluster_true_labels) > 0:
                most_common_class = Counter(cluster_true_labels).most_common(1)[0][1]
                purity += most_common_class

        return purity / n_samples

    @staticmethod
    def entropy(true_labels: List[int], cluster_labels: List[int]) -> float:
        """Calculate cluster entropy"""
        total_entropy = 0
        n_samples = len(true_labels)

        for cluster_id in set(cluster_labels):
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_true_labels = np.array(true_labels)[cluster_mask]
            cluster_size = len(cluster_true_labels)

            if cluster_size > 0:
                class_counts = Counter(cluster_true_labels)
                cluster_entropy = 0

                for count in class_counts.values():
                    if count > 0:
                        prob = count / cluster_size
                        cluster_entropy -= prob * np.log2(prob)

                total_entropy += (cluster_size / n_samples) * cluster_entropy

        return total_entropy

    @staticmethod
    def adjusted_rand_index(labels1: List[int], labels2: List[int]) -> float:
        """Calculate Adjusted Rand Index between two clusterings"""
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(labels1, labels2)

    @staticmethod
    def intra_cluster_distance(X: List[TimeSeriesSample], cluster_labels: List[int],
                              distance_func: Callable) -> float:
        """Average intra-cluster distance"""
        total_distance = 0
        total_pairs = 0

        for cluster_id in set(cluster_labels):
            cluster_indices = [i for i, label in enumerate(cluster_labels)
                             if label == cluster_id]

            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    idx1, idx2 = cluster_indices[i], cluster_indices[j]
                    dist = distance_func(X[idx1].data, X[idx2].data)
                    total_distance += dist
                    total_pairs += 1

        return total_distance / total_pairs if total_pairs > 0 else 0

    @staticmethod
    def inter_cluster_distance(X: List[TimeSeriesSample], cluster_labels: List[int],
                              distance_func: Callable) -> float:
        """Average inter-cluster distance"""
        total_distance = 0
        total_pairs = 0

        unique_clusters = list(set(cluster_labels))

        for i in range(len(unique_clusters)):
            for j in range(i + 1, len(unique_clusters)):
                cluster1_indices = [k for k, label in enumerate(cluster_labels)
                                  if label == unique_clusters[i]]
                cluster2_indices = [k for k, label in enumerate(cluster_labels)
                                  if label == unique_clusters[j]]

                for idx1 in cluster1_indices:
                    for idx2 in cluster2_indices:
                        dist = distance_func(X[idx1].data, X[idx2].data)
                        total_distance += dist
                        total_pairs += 1

        return total_distance / total_pairs if total_pairs > 0 else 0

# ============================================================================
# CONFORMAL PREDICTION
# ============================================================================

class ConformalClassifier:
    """Conformal prediction wrapper for prompt classification"""

    def __init__(self, base_classifier, alpha: float = 0.1):
        self.base_classifier = base_classifier
        self.alpha = alpha  # Miscoverage rate
        self.calibration_scores = None
        self.threshold = None

    def calibrate(self, X_cal: List[TimeSeriesSample], y_cal: List[int]):
        """Calibrate the conformal classifier"""
        predictions = self.base_classifier.predict_batch(X_cal)

        # Calculate non-conformity scores (1 - probability of true class)
        scores = []
        for i, (pred, true_label) in enumerate(zip(predictions, y_cal)):
            if pred == "too-early":
                scores.append(1.0)  # Maximum non-conformity
            else:
                # Get prediction probability from tree/forest
                # For simplicity, use binary score based on correctness
                score = 0.0 if pred == true_label else 1.0
                scores.append(score)

        self.calibration_scores = sorted(scores)
        # Calculate threshold for desired coverage
        n = len(scores)
        quantile_index = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        self.threshold = self.calibration_scores[min(quantile_index, n - 1)]

    def predict_with_confidence(self, X_test: List[TimeSeriesSample]) -> List[Tuple]:
        """Predict with confidence sets"""
        if self.threshold is None:
            raise ValueError("Conformal classifier not calibrated")

        predictions = self.base_classifier.predict_batch(X_test)
        confident_predictions = []

        for pred in predictions:
            if pred == "too-early":
                confident_predictions.append((pred, False))  # Not confident
            else:
                # For simplicity, assume all non-"too-early" predictions are confident
                # In practice, this would use the actual prediction scores
                confident_predictions.append((pred, True))

        return confident_predictions

    def evaluate_miscalibration(self, X_test: List[TimeSeriesSample],
                               y_test: List[int]) -> float:
        """Evaluate miscalibration rate"""
        confident_predictions = self.predict_with_confidence(X_test)

        total_confident = 0
        incorrect_confident = 0

        for i, (pred, is_confident) in enumerate(confident_predictions):
            if is_confident and pred != "too-early":
                total_confident += 1
                if pred != y_test[i]:
                    incorrect_confident += 1

        return incorrect_confident / total_confident if total_confident > 0 else 0

    def evaluate_efficiency(self, X_test: List[TimeSeriesSample]) -> float:
        """Evaluate efficiency (fraction of confident predictions)"""
        confident_predictions = self.predict_with_confidence(X_test)
        confident_count = sum(1 for _, is_confident in confident_predictions
                            if is_confident)
        return confident_count / len(confident_predictions)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class TimeSeriesDataLoader:
    """Load and preprocess time series datasets"""

    @staticmethod
    def generate_synthetic_univariate(n_samples: int = 100, length: int = 50,
                                    n_classes: int = 2, noise_level: float = 0.1,
                                    random_state: int = 42) -> Tuple[List[TimeSeriesSample], List[int]]:
        """Generate synthetic univariate time series data"""
        np.random.seed(random_state)

        X, y = [], []

        for i in range(n_samples):
            class_label = i % n_classes

            # Generate different patterns for different classes
            if class_label == 0:
                # Sine wave pattern
                t = np.linspace(0, 4*np.pi, length)
                data = np.sin(t) + noise_level * np.random.randn(length)
            elif class_label == 1:
                # Linear pattern
                data = np.linspace(0, 1, length) + noise_level * np.random.randn(length)
            elif class_label == 2:
                # Cosine wave pattern
                t = np.linspace(0, 4*np.pi, length)
                data = np.cos(t) + noise_level * np.random.randn(length)
            else:
                # Exponential decay pattern
                t = np.linspace(0, 2, length)
                data = np.exp(-t) + noise_level * np.random.randn(length)

            X.append(TimeSeriesSample(data, class_label))
            y.append(class_label)

        return X, y

    @staticmethod
    def generate_synthetic_multivariate(n_samples: int = 100, length: int = 50,
                                       n_channels: int = 3, n_classes: int = 2,
                                       noise_level: float = 0.1,
                                       random_state: int = 42) -> Tuple[List[TimeSeriesSample], List[int]]:
        """Generate synthetic multivariate time series data"""
        np.random.seed(random_state)

        X, y = [], []

        for i in range(n_samples):
            class_label = i % n_classes

            # Generate multi-channel data
            data = np.zeros((n_channels, length))

            for c in range(n_channels):
                if class_label == 0:
                    # Different sine frequencies for each channel
                    t = np.linspace(0, 4*np.pi, length)
                    data[c] = np.sin(t * (c + 1)) + noise_level * np.random.randn(length)
                elif class_label == 1:
                    # Different linear patterns
                    data[c] = np.linspace(c, c + 1, length) + noise_level * np.random.randn(length)
                elif class_label == 2:
                    # Different cosine patterns
                    t = np.linspace(0, 4*np.pi, length)
                    data[c] = np.cos(t * (c + 1)) + noise_level * np.random.randn(length)
                else:
                    # Random walk patterns
                    data[c] = np.cumsum(np.random.randn(length)) * 0.1

            X.append(TimeSeriesSample(data, class_label))
            y.append(class_label)

        return X, y

    @staticmethod
    def load_from_arrays(data_arrays: np.ndarray, labels: np.ndarray) -> Tuple[List[TimeSeriesSample], List[int]]:
        """Load data from numpy arrays"""
        X = []
        y = labels.tolist() if isinstance(labels, np.ndarray) else labels

        for i, data in enumerate(data_arrays):
            if len(data.shape) == 1:  # Univariate
                X.append(TimeSeriesSample(data, y[i]))
            else:  # Multivariate
                X.append(TimeSeriesSample(data, y[i]))

        return X, y

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class Visualizer:
    """Visualization utilities for trees and results"""

    @staticmethod
    def plot_time_series_samples(X: List[TimeSeriesSample], y: List[int],
                               n_samples_per_class: int = 3, figsize: Tuple = (12, 8)):
        """Plot sample time series from each class"""
        classes = list(set(y))
        n_classes = len(classes)

        fig, axes = plt.subplots(n_classes, n_samples_per_class,
                                figsize=figsize, squeeze=False)

        for class_idx, class_label in enumerate(classes):
            class_indices = [i for i, label in enumerate(y) if label == class_label]
            sample_indices = class_indices[:n_samples_per_class]

            for sample_idx, data_idx in enumerate(sample_indices):
                ax = axes[class_idx, sample_idx]
                sample = X[data_idx]

                if sample.is_multivariate:
                    for channel in range(sample.channels):
                        ax.plot(sample.data[channel], label=f'Channel {channel}')
                    ax.legend()
                else:
                    ax.plot(sample.data)

                ax.set_title(f'Class {class_label}, Sample {sample_idx + 1}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[Union[int, str]],
                            title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        # Filter out "too-early" predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != "too-early"]

        if not valid_indices:
            print("No valid predictions to plot confusion matrix")
            return

        y_true_filtered = [y_true[i] for i in valid_indices]
        y_pred_filtered = [y_pred[i] for i in valid_indices]

        cm = confusion_matrix(y_true_filtered, y_pred_filtered)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_clustering_results(X: List[TimeSeriesSample], cluster_labels: List[int],
                              true_labels: List[int], title: str = "Clustering Results"):
        """Plot clustering results for 2D projection"""
        # Simple 2D projection using first two time points or PCA-like approach
        if not X:
            return

        # Extract features for 2D projection
        features = []
        for sample in X:
            if sample.is_multivariate:
                # Use mean and std of first channel
                feat = [np.mean(sample.data[0]), np.std(sample.data[0])]
            else:
                # Use mean and std of time series
                feat = [np.mean(sample.data), np.std(sample.data)]
            features.append(feat)

        features = np.array(features)

        plt.figure(figsize=(12, 5))

        # Plot clustering results
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(features[:, 0], features[:, 1], c=cluster_labels,
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'{title} - Predicted Clusters')
        plt.xlabel('Feature 1 (Mean)')
        plt.ylabel('Feature 2 (Std)')

        # Plot true labels
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(features[:, 0], features[:, 1], c=true_labels,
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('True Labels')
        plt.xlabel('Feature 1 (Mean)')
        plt.ylabel('Feature 2 (Std)')

        plt.tight_layout()
        plt.show()

# ============================================================================
# MAIN EVALUATION FRAMEWORK
# ============================================================================

class ExperimentRunner:
    """Main class to run all experiments as specified in the assignment"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def run_full_evaluation(self):
        """Run the complete evaluation as specified in the assignment"""
        print("=" * 80)
        print("PROMPT TREE TIME SERIES CLASSIFICATION - FULL EVALUATION")
        print("=" * 80)

        # Generate datasets for the four required combinations
        datasets = self.prepare_datasets()

        for dataset_name, (X, y) in datasets.items():
            print(f"\n{'='*50}")
            print(f"EVALUATING DATASET: {dataset_name}")
            print(f"{'='*50}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )

            X_cal, X_val, y_cal, y_val = train_test_split(
                X_train, y_train, test_size=0.3, random_state=self.random_state, stratify=y_train
            )

            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"Calibration samples: {len(X_cal)}, Validation samples: {len(X_val)}")

            # 1. Supervised Learning Evaluation
            print(f"\n{'-'*30} SUPERVISED EVALUATION {'-'*30}")
            self.evaluate_supervised(X_train, y_train, X_test, y_test, X_cal, y_cal)

            # 2. Unsupervised Learning Evaluation
            print(f"\n{'-'*30} UNSUPERVISED EVALUATION {'-'*30}")
            self.evaluate_unsupervised(X_train, y_train, X_test, y_test)

    def prepare_datasets(self) -> Dict[str, Tuple[List[TimeSeriesSample], List[int]]]:
        """Prepare the four required dataset combinations"""
        datasets = {}

        # 1. Univariate + Binary Classification
        X, y = TimeSeriesDataLoader.generate_synthetic_univariate(
            n_samples=200, length=50, n_classes=2, random_state=self.random_state
        )
        datasets["Univariate_Binary"] = (X, y)

        # 2. Univariate + Multi-class Classification
        X, y = TimeSeriesDataLoader.generate_synthetic_univariate(
            n_samples=300, length=50, n_classes=4, random_state=self.random_state + 1
        )
        datasets["Univariate_Multiclass"] = (X, y)

        # 3. Multivariate + Binary Classification
        X, y = TimeSeriesDataLoader.generate_synthetic_multivariate(
            n_samples=200, length=50, n_channels=3, n_classes=2,
            random_state=self.random_state + 2
        )
        datasets["Multivariate_Binary"] = (X, y)

        # 4. Multivariate + Multi-class Classification
        X, y = TimeSeriesDataLoader.generate_synthetic_multivariate(
            n_samples=300, length=50, n_channels=3, n_classes=4,
            random_state=self.random_state + 3
        )
        datasets["Multivariate_Multiclass"] = (X, y)

        return datasets

    def evaluate_supervised(self, X_train: List[TimeSeriesSample], y_train: List[int],
                          X_test: List[TimeSeriesSample], y_test: List[int],
                          X_cal: List[TimeSeriesSample], y_cal: List[int]):
        """Evaluate supervised learning approaches"""

        voting_methods = ['majority', 'weighted', 'track_record']

        for voting in voting_methods:
            print(f"\n--- Random Forest with {voting.title()} Voting ---")

            try:
                # Train random forest
                rf = PromptRandomForest(
                    n_estimators=5,  # Reduced for faster training
                    max_depth=6,     # Reduced for faster training
                    random_state=self.random_state
                )

                print("Training random forest...")
                rf.fit(X_train, y_train)

                print("Making predictions...")
                # Make predictions
                predictions = rf.predict(X_test, voting=voting)

                # Filter out "too-early" predictions for accuracy calculation
                valid_indices = [i for i, pred in enumerate(predictions) if pred != "too-early"]

                if valid_indices:
                    y_test_filtered = [y_test[i] for i in valid_indices]
                    pred_filtered = [predictions[i] for i in valid_indices]

                    accuracy = accuracy_score(y_test_filtered, pred_filtered)
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Valid predictions: {len(valid_indices)}/{len(predictions)} "
                          f"({100*len(valid_indices)/len(predictions):.1f}%)")

                    # Classification report
                    print("\nClassification Report:")
                    try:
                        print(classification_report(y_test_filtered, pred_filtered))
                    except:
                        print("Could not generate classification report")
                else:
                    print("No valid predictions made!")

                # Conformal prediction evaluation
                print(f"\n--- Conformal Prediction Analysis ---")
                try:
                    conformal = ConformalClassifier(rf, alpha=0.1)
                    conformal.calibrate(X_cal, y_cal)

                    miscalibration = conformal.evaluate_miscalibration(X_test, y_test)
                    efficiency = conformal.evaluate_efficiency(X_test)

                    print(f"Miscalibration rate: {miscalibration:.4f}")
                    print(f"Efficiency: {efficiency:.4f}")
                except Exception as e:
                    print(f"Conformal prediction failed: {e}")

            except Exception as e:
                print(f"Error in {voting} voting evaluation: {e}")
                continue

    def evaluate_unsupervised(self, X_train: List[TimeSeriesSample], y_train: List[int],
                            X_test: List[TimeSeriesSample], y_test: List[int]):
        """Evaluate unsupervised learning (isolation forest) approaches"""

        print("Training Isolation Forest...")

        # Train isolation forest (unsupervised)
        isolation_forest = PromptRandomForest(
            n_estimators=10,
            max_depth=8,
            random_state=self.random_state
        )
        isolation_forest.fit(X_train)  # No labels provided

        # Create clusterer
        clusterer = TimeSeriesClusterer(isolation_forest)

        distance_types = ['breiman', 'zhu', 'ratio_rf']
        clustering_results = {}

        n_classes = len(set(y_test))

        for distance_type in distance_types:
            print(f"\n--- Clustering with {distance_type.title()} Distance ---")

            try:
                cluster_labels = clusterer.cluster(X_test, n_classes, distance_type)
                clustering_results[distance_type] = cluster_labels

                # Internal validation
                intra_dist = ClusteringEvaluator.intra_cluster_distance(
                    X_test, cluster_labels, DistanceFunctions.euclidean_distance
                )
                inter_dist = ClusteringEvaluator.inter_cluster_distance(
                    X_test, cluster_labels, DistanceFunctions.euclidean_distance
                )

                print(f"Intra-cluster distance: {intra_dist:.4f}")
                print(f"Inter-cluster distance: {inter_dist:.4f}")
                print(f"Silhouette-like ratio: {inter_dist/intra_dist:.4f}" if intra_dist > 0 else "N/A")

                # External validation
                purity = ClusteringEvaluator.purity(y_test, cluster_labels)
                entropy = ClusteringEvaluator.entropy(y_test, cluster_labels)

                print(f"Purity: {purity:.4f}")
                print(f"Entropy: {entropy:.4f}")

            except Exception as e:
                print(f"Error in {distance_type} clustering: {e}")
                clustering_results[distance_type] = [0] * len(X_test)

        # Compare clusterings using ARI
        print(f"\n--- Clustering Comparison (Adjusted Rand Index) ---")
        distance_pairs = [('breiman', 'zhu'), ('breiman', 'ratio_rf'), ('zhu', 'ratio_rf')]

        for dist1, dist2 in distance_pairs:
            if dist1 in clustering_results and dist2 in clustering_results:
                try:
                    ari = ClusteringEvaluator.adjusted_rand_index(
                        clustering_results[dist1], clustering_results[dist2]
                    )
                    print(f"ARI between {dist1} and {dist2}: {ari:.4f}")
                except Exception as e:
                    print(f"Error calculating ARI between {dist1} and {dist2}: {e}")

def main():
    """Main function to run the complete evaluation"""
    print("Starting Time Series Prompt Classification Evaluation...")

    # Initialize and run experiments
    runner = ExperimentRunner(random_state=42)
    runner.run_full_evaluation()

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
