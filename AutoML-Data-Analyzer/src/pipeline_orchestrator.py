"""
Orchestrator for Complete ML Pipeline.

Integrates:
- Data validation & cleaning
- Preprocessing
- Clustering
- Modeling
- Insight generation

Entry point for production ML workflow.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from src.data_processing.data_validation import DataValidator
from src.data_processing.preprocessing_pipeline import (
    apply_pca,
    preprocess_for_clustering,
    preprocess_for_modeling,
)
from src.clustering.clustering_pipeline import (
    KMeansClusterer,
    DBSCANClusterer,
    ClusteringEvaluator,
)
from src.modeling.modeling_pipeline import MultiModelTrainer
from src.insights.insights_pipeline import InsightGenerator

logger = logging.getLogger(__name__)


class ProductionMLPipeline:
    """
    Production-grade ML pipeline orchestrator.

    Workflow:
    1. Validate & clean data (remove ID columns, constant features)
    2. Preprocess (imputation, scaling, encoding)
    3. Clustering (KMeans + DBSCAN, select best)
    4. Modeling (multi-model training with CV)
    5. Generate insights (statistical + ML-driven)
    """

    def __init__(self, random_state: int = 42, verbose: bool = True):
        """Initialize pipeline."""
        self.random_state = random_state
        self.verbose = verbose
        self.validator = DataValidator()
        self.insight_generator = InsightGenerator()
        self.pipeline_results = {}
        self.removed_features_report = {}

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        run_modeling: bool = False,
        auto_clean: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete ML pipeline.

        Args:
            df: Input dataframe
            target_col: Target column for supervised learning (optional)
            run_modeling: Whether to run predictive modeling (default False)
            auto_clean: Automatically remove problematic features

        Returns:
            Dict containing all results
        """
        logger.info("=" * 60)
        logger.info("STARTING PRODUCTION ML PIPELINE")
        logger.info("=" * 60)

        # STEP 1: Validate & Clean
        logger.info("\n[STEP 1] Data Validation & Cleaning")
        df_clean = self._validate_and_clean(df, target_col, auto_clean)

        # STEP 2: Run Clustering (ALWAYS)
        logger.info("\n[STEP 2] Clustering Analysis")
        clustering_result = self._run_clustering(df_clean, target_col)

        # STEP 3: Run Modeling (OPTIONAL - only if run_modeling=True AND target_col exists)
        modeling_result = None
        if run_modeling and target_col and target_col in df.columns:
            logger.info("\n[STEP 3] Supervised Learning Modeling")
            modeling_result = self._run_modeling(df_clean, target_col)
        else:
            if run_modeling and not target_col:
                logger.info("\n[STEP 3] Supervised Learning: SKIPPED (target column required)")
            else:
                logger.info("\n[STEP 3] Supervised Learning: SKIPPED (not requested)")

        # STEP 4: Generate Insights
        logger.info("\n[STEP 4] Insight Generation")
        insights_result = self._generate_insights(
            df_clean, clustering_result, modeling_result, target_col
        )

        # Compile results
        final_result = {
            "status": "success",
            "clean_data_shape": df_clean.shape,
            "removed_features": self.removed_features_report,
            "clustering": clustering_result,
            "modeling": modeling_result,
            "insights": insights_result,
        }

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE - ALL STEPS SUCCESSFUL")
        logger.info("=" * 60)

        return final_result

    def _validate_and_clean(
        self, df: pd.DataFrame, target_col: Optional[str], auto_clean: bool
    ) -> pd.DataFrame:
        """Validate and clean data."""
        logger.info(f"  Input shape: {df.shape}")
        logger.info(f"  Missing values: {df.isna().sum().sum()}")

        if auto_clean:
            df_clean = self.validator.validate_and_clean(df, target_col, verbose=True)
            self.removed_features_report = self.validator.get_removed_features_report()
            logger.info(f"  Output shape: {df_clean.shape}")
            logger.info(
                f"  Removed {df.shape[1] - df_clean.shape[1]} problematic features"
            )
        else:
            df_clean = df.copy()
            self.removed_features_report = {}
            logger.info("  Auto-cleaning disabled")

        return df_clean

    def _run_clustering(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> Dict[str, Any]:
        """Run clustering pipeline."""

        # Prepare data (exclude target if present)
        exclude_cols = []
        if target_col and target_col in df.columns:
            exclude_cols = [target_col]

        X_scaled, df_processed, preprocessor = preprocess_for_clustering(
            df, exclude_cols=exclude_cols, target_col=target_col
        )

        logger.info(f"  Preprocessed shape: {X_scaled.shape}")

        # Run KMeans
        logger.info("  Training KMeans...")
        kmeans = KMeansClusterer(random_state=self.random_state)
        kmeans_result = kmeans.fit(X_scaled)

        # Run DBSCAN
        logger.info("  Training DBSCAN...")
        dbscan = DBSCANClusterer()
        dbscan_result = dbscan.fit(X_scaled)

        # Select best
        logger.info("  Evaluating clustering models...")
        results = {"KMeans": kmeans_result, "DBSCAN": dbscan_result}

        best_model_name, best_result = ClusteringEvaluator.select_best_model(results)

        logger.info(f"  Best model: {best_model_name}")
        logger.info(
            f"  Clusters: {best_result.n_clusters}, Silhouette: {best_result.silhouette_score:.4f}"
        )

        # Add labels to processed dataframe (CRITICAL: use df_processed, not raw df)
        df_with_clusters = df_processed.copy()
        df_with_clusters["cluster"] = best_result.labels

        # Compute PCA for visualization only.
        pca_df = apply_pca(df_processed, n_components=2)
        pca_data = pca_df.to_numpy() if pca_df is not None else None
        if pca_df is None:
            logger.warning("  PCA visualization data unavailable for this dataset")

        return {
            "best_model": best_model_name,
            "result": best_result,
            "all_results": results,
            "df_with_clusters": df_with_clusters,
            "evaluation": ClusteringEvaluator.get_evaluation_summary(best_result),
            "pca_data": pca_data,
            "X_processed": X_scaled,
            "labels": best_result.labels,
            "algorithm": best_model_name,
            "preprocessed_data": df_processed,
        }

    def _run_modeling(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Run supervised learning pipeline."""

        # Detect and remove problematic columns
        cols_to_remove = self.validator.detect_problematic_features(df, target_col)[0]

        # Prepare data
        X, y, df_processed, original_features, _preprocessor = preprocess_for_modeling(
            df, target_col, exclude_cols=cols_to_remove, use_knn=False
        )

        logger.info(f"  Features shape after preprocessing: {X.shape}")
        logger.info(f"  Target samples: {len(y)}")

        # Train models
        logger.info("  Training multiple models with 5-fold CV...")
        trainer = MultiModelTrainer(n_splits=5, random_state=self.random_state)
        modeling_result = trainer.train(X, y, original_features)

        logger.info(f"  Best model: {modeling_result.best_model_name}")
        logger.info(f"  Best score: {modeling_result.best_score:.4f}")
        logger.info(f"  Problem type: {modeling_result.problem_type}")

        return {
            "best_model_name": modeling_result.best_model_name,
            "best_score": modeling_result.best_score,
            "all_scores": modeling_result.all_scores,
            "feature_importance": modeling_result.feature_importance,
            "original_features": modeling_result.original_features,
            "cross_val_scores": modeling_result.cross_val_scores,
            "metrics": modeling_result.metrics,
            "problem_type": modeling_result.problem_type,
            "training_data": modeling_result.training_data,
            "performance_band": modeling_result.performance_band,
            "processed_features": df_processed,
        }

    def _generate_insights(
        self,
        df: pd.DataFrame,
        clustering_result: Dict[str, Any],
        modeling_result: Optional[Dict[str, Any]],
        target_col: Optional[str],
    ) -> Dict[str, Any]:
        """Generate comprehensive insights."""

        # Get cluster labels if available
        cluster_labels = (
            clustering_result["df_with_clusters"]["cluster"].values
            if "df_with_clusters" in clustering_result
            else None
        )
        silhouette_score = (
            clustering_result["result"].silhouette_score
            if "result" in clustering_result
            else None
        )

        # Get modeling info if available
        feature_importance = None
        best_model_name = None
        best_score = None
        problem_type = "unknown"
        original_features = None

        if modeling_result:
            feature_importance = modeling_result.get("feature_importance", {})
            best_model_name = modeling_result.get("best_model_name")
            best_score = modeling_result.get("best_score")
            problem_type = modeling_result.get("problem_type", "unknown")
            original_features = modeling_result.get("original_features", [])

        # Generate insights
        insights = self.insight_generator.generate_full_report(
            df,
            cluster_labels=cluster_labels,
            silhouette_score=silhouette_score,
            feature_importance=feature_importance,
            best_model_name=best_model_name,
            best_score=best_score,
            problem_type=problem_type,
            original_features=original_features,
        )

        return {
            "feature_statistics": insights.feature_statistics,
            "clustering_insights": insights.clustering_insights,
            "model_insights": insights.model_insights,
            "recommendations": insights.recommendations,
            "quality_metrics": insights.quality_metrics,
        }


# Convenience functions
def run_complete_pipeline(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    run_modeling: bool = False,
    auto_clean: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run complete pipeline.

    Args:
        df: Input dataframe
        target_col: Optional target column
        run_modeling: Whether to run predictive modeling (default False)
        auto_clean: Whether to auto-clean data

    Returns:
        Complete pipeline results
    """
    pipeline = ProductionMLPipeline(verbose=True)
    return pipeline.run_full_pipeline(df, target_col, run_modeling, auto_clean)
