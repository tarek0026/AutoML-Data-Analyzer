"""
Production-Grade Clustering Pipeline.

Implements:
- Proper scaling before clustering
- Automatic K selection for KMeans
- Eps/min_samples estimation for DBSCAN
- Unified evaluation metric
- Best model selection
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    model_name: str
    labels: np.ndarray
    model: Any
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    n_clusters: int
    n_noise: int
    metadata: Dict[str, Any]


class KMeansClusterer:
    """Production-grade KMeans clustering with automatic K selection."""
    
    def __init__(self, max_k: int = 10, min_k: int = 2, random_state: int = 42):
        """
        Initialize KMeans clusterer.
        
        Args:
            max_k: Maximum number of clusters to try
            min_k: Minimum number of clusters
            random_state: Random seed
        """
        self.max_k = max_k
        self.min_k = min_k
        self.random_state = random_state
        self.model = None
        self.scaler = None
    
    def estimate_optimal_k(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Estimate optimal K using silhouette score.
        
        Args:
            X: Input array (will be scaled internally)
        
        Returns:
            Tuple of (best_k, best_silhouette_score)
        """
        logger.info(f"Estimating optimal K (range: {self.min_k}-{self.max_k})")
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        best_k = self.min_k
        best_score = -2  # Silhouette ranges from -1 to 1
        
        for k in range(self.min_k, self.max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, 
                               n_init=10, max_iter=300)
                labels = kmeans.fit_predict(X_scaled)
                
                # Need at least 2 clusters for silhouette
                if len(np.unique(labels)) < 2:
                    continue
                
                score = silhouette_score(X_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                
                logger.debug(f"  K={k}: silhouette_score={score:.4f}")
            
            except Exception as e:
                logger.warning(f"  K={k}: Failed - {e}")
                continue
        
        logger.info(f"Optimal K={best_k} with silhouette_score={best_score:.4f}")
        return best_k, best_score
    
    def fit(self, X: np.ndarray, k: Optional[int] = None) -> ClusteringResult:
        """
        Fit KMeans model.
        
        Args:
            X: Input array
            k: Number of clusters (auto if None)
        
        Returns:
            ClusteringResult object
        """
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Estimate K if not provided
        if k is None:
            k, silhouette_est = self.estimate_optimal_k(X)
        else:
            silhouette_est = None
        
        # Fit model
        self.model = KMeans(n_clusters=k, random_state=self.random_state,
                           n_init=10, max_iter=300)
        labels = self.model.fit_predict(X_scaled)
        
        silhouette, davies_bouldin, calinski_harabasz = _safe_clustering_metrics(X_scaled, labels)
        
        result = ClusteringResult(
            model_name='KMeans',
            labels=labels,
            model=self.model,
            silhouette_score=silhouette,
            davies_bouldin_score=davies_bouldin,
            calinski_harabasz_score=calinski_harabasz,
            n_clusters=k,
            n_noise=0,
            metadata={
                'k': k,
                'inertia': float(self.model.inertia_),
                'silhouette_estimation': silhouette_est
            }
        )
        
        logger.info(f"KMeans fit: K={k}, silhouette={silhouette:.4f}")
        return result


class DBSCANClusterer:
    """Production-grade DBSCAN clustering with automatic eps estimation."""
    
    def __init__(self, min_samples: Optional[int] = None, percentile: int = 90):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            min_samples: Minimum samples per cluster (auto if None)
            percentile: Percentile for eps estimation (90th typical)
        """
        self.min_samples = min_samples
        self.percentile = percentile
        self.model = None
        self.scaler = None
    
    def estimate_eps(self, X: np.ndarray, k: int = 5) -> float:
        """
        Estimate eps parameter using k-distance graph.
        
        Args:
            X: Input array (assumed scaled)
            k: Number of neighbors for k-distance calculation
        
        Returns:
            Estimated eps value
        """
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        
        # Sort distances to k-th neighbor
        k_distances = np.sort(distances[:, k - 1])
        eps = np.percentile(k_distances, self.percentile)
        
        logger.debug(f"Estimated eps={eps:.4f} at {self.percentile}th percentile")
        return float(eps)
    
    def fit(self, X: np.ndarray, eps: Optional[float] = None,
            min_samples: Optional[int] = None) -> ClusteringResult:
        """
        Fit DBSCAN model.
        
        Args:
            X: Input array
            eps: Eps parameter (auto if None)
            min_samples: Min samples parameter (auto if None)
        
        Returns:
            ClusteringResult object
        """
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Auto estimate min_samples
        if min_samples is None:
            min_samples = max(5, int(np.log(len(X))))
        
        # Auto estimate eps
        if eps is None:
            eps = self.estimate_eps(X_scaled, k=min_samples)
        
        # Fit model
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Only calculate silhouette if valid clusters
        silhouette, davies_bouldin, calinski_harabasz = _safe_clustering_metrics(X_scaled, labels)
        
        result = ClusteringResult(
            model_name='DBSCAN',
            labels=labels,
            model=self.model,
            silhouette_score=silhouette,
            davies_bouldin_score=davies_bouldin,
            calinski_harabasz_score=calinski_harabasz,
            n_clusters=n_clusters,
            n_noise=n_noise,
            metadata={
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
        )
        
        logger.info(f"DBSCAN fit: clusters={n_clusters}, noise={n_noise}, "
                   f"silhouette={silhouette:.4f}")
        return result


class ClusteringEvaluator:
    """Evaluate and compare clustering results."""
    
    @staticmethod
    def select_best_model(results: Dict[str, ClusteringResult]) -> Tuple[str, ClusteringResult]:
        """
        Select best clustering model based on metrics.
        
        Strategy:
        - Prefer higher silhouette score (better separation)
        - Penalize excessive noise (DBSCAN)
        - Penalize single cluster (no clustering)
        
        Args:
            results: Dict of model_name -> ClusteringResult
        
        Returns:
            Tuple of (best_model_name, best_result)
        """
        best_model_name = None
        best_score = -np.inf
        best_result = None
        
        for model_name, result in results.items():
            # Base score: silhouette (prefer higher)
            score = result.silhouette_score
            
            # Penalize if only 1 cluster
            if result.n_clusters <= 1:
                score -= 1.0
            
            # Penalize excessive noise (> 30%)
            noise_ratio = result.n_noise / len(result.labels) if len(result.labels) > 0 else 0
            if noise_ratio > 0.3:
                score -= (noise_ratio - 0.3) * 2
            
            logger.debug(f"{model_name}: raw_silhouette={result.silhouette_score:.4f}, "
                        f"adjusted_score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_result = result
        
        if best_result is None:
            raise ValueError("Could not select best model from results")
        
        logger.info(f"Best model: {best_model_name} (score={best_score:.4f})")
        return best_model_name, best_result
    
    @staticmethod
    def get_evaluation_summary(result: ClusteringResult) -> dict:
        """Get human-readable summary of clustering evaluation."""
        return {
            'model': result.model_name,
            'n_clusters': result.n_clusters,
            'n_noise': result.n_noise,
            'silhouette_score': round(result.silhouette_score, 4),
            'davies_bouldin_score': round(result.davies_bouldin_score, 4),
            'calinski_harabasz_score': round(result.calinski_harabasz_score, 4),
            'quality': 'strong' if result.silhouette_score > 0.5 else
                      'moderate' if result.silhouette_score > 0.2 else 'weak'
        }


def _safe_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Calculate clustering metrics safely for edge cases."""
    unique_labels = np.unique(labels)
    valid_clusters = [label for label in unique_labels if label != -1]

    if len(valid_clusters) < 2:
        return -1.0, np.nan, np.nan

    try:
        return (
            float(silhouette_score(X_scaled, labels)),
            float(davies_bouldin_score(X_scaled, labels)),
            float(calinski_harabasz_score(X_scaled, labels)),
        )
    except Exception as exc:
        logger.warning("Could not compute clustering metrics safely: %s", exc)
        return -1.0, np.nan, np.nan
