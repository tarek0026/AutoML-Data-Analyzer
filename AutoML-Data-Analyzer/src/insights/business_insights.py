"""
Business-Oriented Insights Generation.

Transforms raw data statistics into actionable business insights.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BusinessInsightGenerator:
    """Generate business-ready, actionable insights."""
    
    @staticmethod
    def generate_data_quality_insights(df: pd.DataFrame) -> List[str]:
        """Generate insights about data quality."""
        insights = []
        
        # Overall record count
        n_records = len(df)
        if n_records < 100:
            insights.append(f"⚠️ **Small Dataset**: Only {n_records} records. Consider collecting more data for reliable patterns.")
        elif n_records < 1000:
            insights.append(f"📊 **Adequate Sample Size**: {n_records:,} records provide good statistical foundation.")
        else:
            insights.append(f"✅ **Robust Dataset**: {n_records:,} records enable reliable analysis and modeling.")
        
        # Missing values
        missing_count = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (missing_count / total_cells) * 100
        
        if missing_pct > 30:
            insights.append(f"⚠️ **High Missing Data**: {missing_pct:.1f}% of data is missing. Data quality issues may impact model reliability.")
        elif missing_pct > 10:
            insights.append(f"⚠️ **Moderate Missing Data**: {missing_pct:.1f}% missing. Review collection processes.")
        elif missing_pct > 0:
            insights.append(f"✅ **Good Data Completeness**: Only {missing_pct:.1f}% missing values.")
        else:
            insights.append(f"✅ **Perfect Data**: No missing values detected.")
        
        # Feature count
        n_features = df.shape[1]
        if n_features < 3:
            insights.append(f"⚠️ **Limited Features**: Only {n_features} features. Consider feature engineering.")
        elif n_features < 10:
            insights.append(f"✅ **Focused Feature Set**: {n_features} features provide clear analysis scope.")
        else:
            insights.append(f"📊 **Rich Data**: {n_features} features enable comprehensive analysis.")
        
        return insights
    
    @staticmethod
    def generate_clustering_insights(n_clusters: int, silhouette_score: float,
                                    cluster_sizes: Dict[int, int],
                                    noise_count: int = 0) -> List[str]:
        """Generate business insights from clustering."""
        insights = []

        silhouette_score = float(silhouette_score) if silhouette_score is not None else -1.0
        
        # Cluster quality
        if silhouette_score > 0.5:
            quality = "**Strong**"
            action = "Segments are reasonably distinct and likely usable."
        elif silhouette_score > 0.2:
            quality = "**Moderate**"
            action = "Segments exist, but overlap suggests cautious interpretation."
        elif silhouette_score > 0:
            quality = "**Weak**"
            action = "Segments are loose and should not be treated as strong business groups yet."
        else:
            quality = "**Weak**"
            action = "No reliable segmentation signal detected. Revisit features or clustering settings."
        
        insights.append(f"🎯 **Cluster Quality**: {quality} ({silhouette_score:.3f}). {action}")
        
        # Number of segments
        if n_clusters == 1:
            insights.append(f"⚠️ **Homogeneous Data**: Only 1 cluster found. Dataset may lack natural groupings.")
        elif n_clusters <= 3:
            insights.append(f"🎯 **Clear Segmentation**: Data naturally groups into **{n_clusters}** distinct segments. Large actionable groups.")
        elif n_clusters <= 5:
            insights.append(f"📊 **Moderate Complexity**: **{n_clusters}** clusters indicate diverse patterns. Requires targeted strategies per segment.")
        elif n_clusters <= 10:
            insights.append(f"🔍 **High Granularity**: **{n_clusters}** clusters. Detailed segmentation enables specialized targeting.")
        else:
            insights.append(f"🔬 **Fine-Grained**: **{n_clusters}** clusters. Consider consolidation for operational simplicity.")
        
        # Size balance
        if cluster_sizes:
            max_size = max(cluster_sizes.values())
            min_size = min(cluster_sizes.values())
            ratio = max_size / min_size if min_size > 0 else 0
            
            if ratio > 10:
                insights.append(f"⚠️ **Unbalanced Segments**: Largest cluster is {ratio:.0f}x larger than smallest. Some segments may be underserved.")
            elif ratio > 3:
                insights.append(f"📊 **Varied Segment Sizes**: Clusters range from {min_size} to {max_size} samples. Tailor strategies per size.")
            else:
                insights.append(f"✅ **Balanced Segments**: Clusters are evenly distributed, enabling consistent strategies.")
        
        # Noise points
        if noise_count > 0:
            noise_pct = (noise_count / (sum(cluster_sizes.values()) + noise_count)) * 100 if cluster_sizes else 0
            if noise_pct > 20:
                insights.append(f"⚠️ **High Outlier Count**: {noise_pct:.1f}% of data are outliers. Investigate anomalies.")
            elif noise_pct > 5:
                insights.append(f"📌 **Outliers Present**: {noise_pct:.1f}% outliers detected. Review for anomalies or special cases.")
        
        return insights
    
    @staticmethod
    def generate_model_insights(best_score: float, problem_type: str,
                               cross_val_scores: Optional[Dict] = None,
                               best_model: str = "") -> List[str]:
        """Generate business insights from model performance."""
        insights = []
        
        if problem_type == 'classification':
            if best_score > 0.75:
                strength = "**Strong** (>75%)"
                reliability = "Predictions have meaningful signal."
                action = "✅ Suitable for guided decision support with monitoring."
            elif best_score >= 0.60:
                strength = "**Moderate** (60%-75%)"
                reliability = "Predictions may help, but uncertainty remains material."
                action = "⚠️ Use with human review and keep improving features/data."
            else:
                strength = "**Weak** (<60%)"
                reliability = "Predictions are unreliable."
                action = "🔴 Not production-ready. Improve preprocessing, features, or data quality first."
            
            insights.append(f"📈 **Model Performance**: {strength} accuracy ({best_score:.1%}). {reliability} {action}")
        
        else:  # regression
            if best_score > 0.75:
                strength = "**Strong** (>0.75 R²)"
                reliability = "Model captures most of the target variation."
                action = "✅ Reasonable candidate for monitored production use."
            elif best_score >= 0.60:
                strength = "**Moderate** (0.60-0.75 R²)"
                reliability = "Model captures some useful signal."
                action = "⚠️ Use for guidance and continue iterating."
            else:
                strength = "**Weak** (<0.60 R²)"
                reliability = "Predictive power is limited."
                action = "🔴 Not production-ready. Requires better features or more data."
            
            insights.append(f"📊 **Model Performance**: {strength} (R² = {best_score:.3f}). {reliability} {action}")
        
        # Cross-validation stability
        if cross_val_scores and best_model in cross_val_scores:
            scores = cross_val_scores[best_model]
            cv_mean = np.mean(scores)
            cv_std = np.std(scores)
            cv_coef = cv_std / cv_mean if cv_mean > 0 else 0
            
            if cv_coef < 0.05:
                insights.append(f"✅ **Stable Model**: Consistent performance across folds (±{cv_std:.3f}). Safe for deployment.")
            elif cv_coef < 0.10:
                insights.append(f"📊 **Reasonable Stability**: Small variance across folds (±{cv_std:.3f}). Generally stable.")
            else:
                insights.append(f"⚠️ **High Variance**: Performance varies significantly (±{cv_std:.3f}). Monitor in production.")
        
        return insights
    
    @staticmethod
    def generate_feature_insights(feature_importance: Dict[str, float],
                                 original_features: List[str],
                                 top_n: int = 3) -> List[str]:
        """Generate insights about top features."""
        insights = []
        
        if not feature_importance:
            insights.append("📊 **Feature Analysis**: Model does not support feature importance extraction.")
            return insights
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        total_importance = sum(feature_importance.values())
        
        # Top drivers
        top_names = [f[0] for f in top_features]
        top_pcts = [f[1] / total_importance * 100 if total_importance > 0 else 0 for f in top_features]
        
        if len(top_names) == 1:
            insights.append(f"🎯 **Primary Driver**: **{top_names[0]}** is the dominant predictor ({top_pcts[0]:.0f}% importance). Focus optimization here.")
        elif len(top_names) == 2:
            insights.append(f"🎯 **Key Drivers**: **{top_names[0]}** and **{top_names[1]}** drive predictions ({top_pcts[0]:.0f}% + {top_pcts[1]:.0f}%).")
        else:
            combined_pct = sum(top_pcts)
            insights.append(f"🎯 **Top Drivers**: **{top_names[0]}**, **{top_names[1]}**, and **{top_names[2]}** account for {combined_pct:.0f}% of model influence.")
        
        # Feature concentration
        top_5_importance = sum([imp for _, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]])
        top_5_pct = top_5_importance / total_importance * 100 if total_importance > 0 else 0
        
        if top_5_pct > 80:
            insights.append(f"🔍 **Concentrated Dependencies**: Top 5 features drive {top_5_pct:.0f}% of predictions. Few key factors matter.")
        elif top_5_pct > 60:
            insights.append(f"📊 **Moderate Dependencies**: Distributed influence across multiple features.")
        else:
            insights.append(f"📈 **Diverse Model**: Many features contribute meaningfully. Robust to individual feature changes.")
        
        return insights
    
    @staticmethod
    def generate_recommendations(problem_type: str, best_score: float,
                               n_records: int, missing_pct: float,
                               feature_importance: Optional[Dict] = None) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if problem_type == 'classification' and best_score < 0.60:
            recommendations.append("📝 **Improve Model Accuracy**: Collect more data, engineer new features, or try ensemble methods.")
        elif problem_type == 'regression' and best_score < 0.60:
            recommendations.append("📝 **Enhance Predictions**: Focus on feature quality and domain-specific engineering.")
        
        # Data recommendations
        if n_records < 1000:
            recommendations.append("📊 **Expand Dataset**: Current size limits pattern detection. Aim for 5,000+ samples for robust insights.")
        
        if missing_pct > 10:
            recommendations.append("🔍 **Review Data Collection**: High missing rate suggests gaps in collection process.")
        
        # Feature-based recommendations
        if feature_importance:
            top_feature = list(feature_importance.items())[0][0]
            recommendations.append(f"💡 **Focus on {top_feature}**: This is your strongest predictor. Invest in improving its quality/measurement.")
        
        # Monitoring recommendations
        if best_score > 0.75:
            recommendations.append("✅ **Set up Monitoring**: Track model performance in production. Retrain monthly or on data drift.")
        else:
            recommendations.append("⚠️ **Plan for Iteration**: Expect model updates as you collect more data. Don't rely solely on current model.")
        
        return recommendations
    
    @staticmethod
    def summarize_analysis(results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate complete insight summary."""
        summary = {
            'data_quality': [],
            'clustering': [],
            'modeling': [],
            'features': [],
            'recommendations': []
        }
        
        # Data quality
        if 'clean_data_shape' in results:
            n_rows, n_cols = results['clean_data_shape']
            df_temp = pd.DataFrame(np.random.randn(n_rows, n_cols))
            summary['data_quality'] = BusinessInsightGenerator.generate_data_quality_insights(df_temp)
        
        # Clustering insights
        if 'clustering' in results and results['clustering']:
            cluster_info = results['clustering'].get('evaluation', {})
            summary['clustering'] = BusinessInsightGenerator.generate_clustering_insights(
                n_clusters=cluster_info.get('n_clusters', 0),
                silhouette_score=cluster_info.get('silhouette_score', 0),
                cluster_sizes={},
                noise_count=cluster_info.get('n_noise', 0)
            )
        
        # Model insights
        if 'modeling' in results and results['modeling']:
            model_info = results['modeling']
            summary['modeling'] = BusinessInsightGenerator.generate_model_insights(
                best_score=model_info.get('best_score', 0),
                problem_type=model_info.get('problem_type', 'unknown'),
                cross_val_scores=model_info.get('cross_val_scores', {}),
                best_model=model_info.get('best_model_name', '')
            )
        
        # Feature insights
        if 'modeling' in results and results['modeling']:
            model_info = results['modeling']
            summary['features'] = BusinessInsightGenerator.generate_feature_insights(
                feature_importance=model_info.get('feature_importance', {}),
                original_features=model_info.get('original_features', [])
            )
        
        # Recommendations
        clean_shape = results.get('clean_data_shape', (0, 0))
        modeling_info = results.get('modeling') or {}
        summary['recommendations'] = BusinessInsightGenerator.generate_recommendations(
            problem_type=modeling_info.get('problem_type', 'unknown'),
            best_score=modeling_info.get('best_score', 0),
            n_records=clean_shape[0],
            missing_pct=0,
            feature_importance=modeling_info.get('feature_importance', {})
        )
        
        return summary
