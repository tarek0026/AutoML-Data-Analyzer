"""
Production-Grade Insight Generation.

Implements:
- Statistical analysis of features
- Clustering behavior interpretation
- Model-driven insights
- Generic, actionable recommendations
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DatasetInsights:
    """Container for dataset-level insights."""
    feature_statistics: Dict[str, Dict[str, Any]]
    clustering_insights: List[str]
    model_insights: List[str]
    recommendations: List[str]
    quality_metrics: Dict[str, Any]


class StatisticalAnalyzer:
    """Analyze statistical properties of features."""
    
    @staticmethod
    def analyze_feature_distribution(series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution of a feature."""
        if pd.api.types.is_numeric_dtype(series):
            clean_data = series.dropna()
            
            if len(clean_data) < 2:
                return {'error': 'Insufficient data'}
            
            return {
                'mean': float(clean_data.mean()),
                'median': float(clean_data.median()),
                'std': float(clean_data.std()),
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'variance': float(clean_data.var()),
                'skewness': float(stats.skew(clean_data)),
                'kurtosis': float(stats.kurtosis(clean_data)),
                'missing_ratio': float(series.isna().sum() / len(series))
            }
        
        else:
            # Categorical
            return {
                'unique_values': int(series.nunique()),
                'mode': series.mode()[0] if len(series.mode()) > 0 else None,
                'missing_ratio': float(series.isna().sum() / len(series)),
                'top_categories': series.value_counts().head(5).to_dict()
            }
    
    @staticmethod
    def identify_outliers(series: pd.Series, method: str = 'iqr') -> Tuple[int, float]:
        """
        Identify outliers in numeric feature.
        
        Args:
            series: Numeric series
            method: 'iqr' (Interquartile Range) or 'zscore'
        
        Returns:
            Tuple of (n_outliers, outlier_ratio)
        """
        clean_data = series.dropna()
        
        if len(clean_data) < 4:
            return 0, 0.0
        
        if method == 'iqr':
            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = (clean_data < lower) | (clean_data > upper)
        
        else:  # zscore
            z_scores = np.abs(stats.zscore(clean_data))
            outliers = z_scores > 3
        
        n_outliers = int(outliers.sum())
        ratio = n_outliers / len(clean_data)
        
        return n_outliers, ratio


class ClusteringInsightGenerator:
    """Generate actionable insights from clustering results."""
    
    @staticmethod
    def analyze_cluster_quality(labels: np.ndarray, silhouette_score: float,
                                n_clusters: int, n_noise: int) -> List[str]:
        """Generate insights about clustering quality."""
        insights = []
        
        # Silhouette score interpretation
        if silhouette_score > 0.5:
            quality = "strong"
            details = "clusters are reasonably well-separated"
        elif silhouette_score > 0.2:
            quality = "moderate"
            details = "clusters show some structure, but overlap is still meaningful"
        elif silhouette_score > 0:
            quality = "weak"
            details = "clusters are loose and should be interpreted cautiously"
        else:
            quality = "weak"
            details = "clusters are not well-defined"
        
        insights.append(
            f"Clustering quality is **{quality}** ({silhouette_score:.3f} silhouette score): "
            f"{details}."
        )
        
        # Cluster count interpretation
        if n_clusters == 1:
            insights.append(
                "⚠️ Warning: Only 1 cluster was identified. "
                "The data may be too homogeneous for meaningful segmentation."
            )
        elif n_clusters <= 3:
            insights.append(
                f"Data naturally groups into **{n_clusters} distinct segments**. "
                f"This level of segmentation suggests clear patterns in the dataset."
            )
        elif n_clusters <= 10:
            insights.append(
                f"Data forms **{n_clusters} clusters**. "
                f"This suggests moderate complexity with multiple but identifiable patterns."
            )
        else:
            insights.append(
                f"Data fragments into **{n_clusters} clusters**. "
                f"This may indicate high diversity or need for dimensionality reduction."
            )
        
        # Noise interpretation
        if n_noise > 0:
            noise_ratio = n_noise / len(labels)
            if noise_ratio > 0.2:
                insights.append(
                    f"⚠️ High noise level ({noise_ratio:.1%} points unassigned). "
                    f"This suggests some anomalous or transitional data points."
                )
            elif noise_ratio > 0.05:
                insights.append(
                    f"{noise_ratio:.1%} of data points are outliers/noise points. "
                    f"These represent boundary cases between clusters."
                )
        
        return insights
    
    @staticmethod
    def generate_cluster_size_insights(df: pd.DataFrame, cluster_col: str = 'cluster') -> List[str]:
        """Generate insights about cluster sizes."""
        if cluster_col not in df.columns:
            return []
        
        insights = []
        cluster_sizes = df[cluster_col].value_counts().sort_values(ascending=False)
        
        # Size distribution
        avg_size = cluster_sizes.mean()
        min_size = cluster_sizes.min()
        max_size = cluster_sizes.max()
        
        if max_size / min_size > 10:
            insights.append(
                f"Cluster sizes vary significantly (largest: {max_size}, smallest: {min_size}). "
                f"Some segments are much more populous than others."
            )
        else:
            insights.append(
                f"Clusters are relatively balanced (avg: {avg_size:.0f} samples). "
                f"Suggests stable, well-distributed patterns."
            )
        
        return insights


class ModelInsightGenerator:
    """Generate insights from trained models."""
    
    @staticmethod
    def generate_feature_insights(feature_importance: Dict[str, float],
                                  original_features: List[str]) -> List[str]:
        """Generate insights about important features."""
        insights = []
        
        if not feature_importance:
            return ["Feature importance could not be computed for this model."]
        
        top_features = list(feature_importance.keys())[:3]
        
        # Top drivers
        if len(top_features) >= 1:
            importance_vals = [feature_importance[f] for f in top_features]
            top_importance_pct = (importance_vals[0] / sum(feature_importance.values())) * 100
            
            features_text = ', '.join([f'**{f}**' for f in top_features[:2]])
            if len(top_features) > 2:
                features_text += f', and **{top_features[2]}**'
            
            insights.append(
                f"**Top predictive drivers**: {features_text}. "
                f"These features are the strongest indicators for model predictions."
            )
        
        # Feature count insight
        total_features = len(original_features)
        important_features = len([f for f, imp in feature_importance.items() if imp > 0])
        
        if important_features < total_features * 0.2:
            insights.append(
                f"Only {important_features}/{total_features} features are significantly influential. "
                f"The model relies on a concentrated set of predictors."
            )
        else:
            insights.append(
                f"{important_features}/{total_features} features contribute meaningfully to predictions. "
                f"The model considers diverse information sources."
            )
        
        return insights
    
    @staticmethod
    def generate_performance_insights(best_score: float, problem_type: str,
                                      best_model_name: str) -> List[str]:
        """Generate insights about model performance."""
        insights = []
        
        if problem_type == 'classification':
            if best_score > 0.75:
                performance = "strong"
                interpretation = "The model has meaningful predictive signal and can support decisions with monitoring"
            elif best_score >= 0.60:
                performance = "moderate"
                interpretation = "The model captures some useful patterns, but predictions still need caution"
            else:
                performance = "weak"
                interpretation = "Model performance is weak. Predictions are unreliable."
            
            insights.append(
                f"Model **{best_model_name}** demonstrates **{performance}** performance "
                f"({best_score:.1%} accuracy). {interpretation}."
            )
        
        else:  # regression (R² score)
            if best_score > 0.75:
                performance = "strong"
                interpretation = "explains most of the variance in the target"
            elif best_score >= 0.60:
                performance = "moderate"
                interpretation = "captures some of the target variance, but errors remain material"
            else:
                performance = "weak"
                interpretation = "captures too little target variance for reliable use"
            
            insights.append(
                f"Model **{best_model_name}** shows **{performance}** predictive power "
                f"(R²={best_score:.3f}), which {interpretation}."
            )
        
        return insights
    
    @staticmethod
    def generate_optimization_insights(feature_importance: Dict[str, float],
                                       best_score: float) -> List[str]:
        """Generate recommendations for model improvement."""
        insights = []
        
        if best_score < 0.6:
            insights.append(
                "💡 **Optimization Opportunity**: Performance is currently weak. "
                "Improve feature quality, revisit preprocessing, or collect more representative data."
            )
        
        if feature_importance:
            top_feature = list(feature_importance.keys())[0]
            insights.append(
                f"**Focus Area**: Improving **{top_feature}** quality/measurement could directly "
                f"enhance model performance, as it's the strongest predictor."
            )
        
        return insights


class InsightGenerator:
    """Main insight generation orchestrator."""
    
    def __init__(self):
        self.stat_analyzer = StatisticalAnalyzer()
        self.cluster_analyzer = ClusteringInsightGenerator()
        self.model_analyzer = ModelInsightGenerator()
    
    def generate_full_report(self,
                           df: pd.DataFrame,
                           cluster_labels: Optional[np.ndarray] = None,
                           silhouette_score: Optional[float] = None,
                           feature_importance: Optional[Dict[str, float]] = None,
                           best_model_name: Optional[str] = None,
                           best_score: Optional[float] = None,
                           problem_type: str = 'unknown',
                           original_features: Optional[List[str]] = None) -> DatasetInsights:
        """
        Generate comprehensive insights.
        
        Args:
            df: Input dataframe (with cluster labels if applicable)
            cluster_labels: Cluster assignments
            silhouette_score: Clustering quality metric
            feature_importance: Top features from model
            best_model_name: Name of best trained model
            best_score: Performance score
            problem_type: 'classification' or 'regression'
            original_features: Original feature names
        
        Returns:
            DatasetInsights object
        """
        # Feature statistics
        feature_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'cluster':  # Exclude cluster column
                feature_stats[col] = self.stat_analyzer.analyze_feature_distribution(df[col])
        
        # Clustering insights
        clustering_insights = []
        if cluster_labels is not None and silhouette_score is not None:
            n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = np.sum(cluster_labels == -1)
            
            clustering_insights.extend(
                self.cluster_analyzer.analyze_cluster_quality(
                    cluster_labels, silhouette_score, n_clusters, n_noise
                )
            )
            
            if 'cluster' in df.columns:
                clustering_insights.extend(
                    self.cluster_analyzer.generate_cluster_size_insights(df, 'cluster')
                )
        
        # Model insights
        model_insights = []
        if feature_importance and best_score is not None:
            model_insights.extend(
                self.model_analyzer.generate_feature_insights(
                    feature_importance, original_features or []
                )
            )
            
            if best_model_name:
                model_insights.extend(
                    self.model_analyzer.generate_performance_insights(
                        best_score, problem_type, best_model_name
                    )
                )
                
                model_insights.extend(
                    self.model_analyzer.generate_optimization_insights(
                        feature_importance, best_score
                    )
                )
        
        # Quality metrics
        quality_metrics = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values_pct': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_metrics, clustering_insights, model_insights
        )
        
        return DatasetInsights(
            feature_statistics=feature_stats,
            clustering_insights=clustering_insights,
            model_insights=model_insights,
            recommendations=recommendations,
            quality_metrics=quality_metrics
        )
    
    def _generate_recommendations(self, quality_metrics: dict,
                                 clustering_insights: List[str],
                                 model_insights: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        if quality_metrics['missing_values_pct'] > 20:
            recommendations.append(
                "**Data Quality**: High missing value ratio. "
                "Consider collecting more complete data or investigate missing patterns."
            )
        
        if quality_metrics['total_samples'] < 100:
            recommendations.append(
                "**Sample Size**: Dataset is small. "
                "Collect more data to improve model robustness and generalization."
            )
        
        if quality_metrics['total_features'] < 3:
            recommendations.append(
                "**Feature Engineering**: Limited feature set. "
                "Engineer new features from existing data to improve model performance."
            )
        
        # Clustering-specific
        if clustering_insights:
            recommendations.append(
                "**Use Case**: Leverage cluster assignments for targeted strategies per segment."
            )
        
        # General recommendation
        if not recommendations:
            recommendations.append(
                "✅ **Good Data Quality**: Dataset appears well-structured. "
                "Consider iterating on feature engineering for further improvements."
            )
        
        return recommendations
