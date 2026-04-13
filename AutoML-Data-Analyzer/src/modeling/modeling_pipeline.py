"""
Production-Grade Modeling Pipeline.

Implements:
- Automatic problem type detection
- Multi-model training with proper cross-validation
- Comprehensive evaluation metrics
- Robust feature importance extraction
- Safe encoding handling
"""

from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

from sklearn.model_selection import cross_validate, StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelingResult:
    """Container for modeling results."""
    problem_type: str  # 'classification' or 'regression'
    best_model_name: str
    best_model: Any
    best_score: float
    all_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    original_features: List[str]
    cross_val_scores: Dict[str, list]
    metrics: Dict[str, Any]
    training_data: Dict[str, Any]
    performance_band: str
    target_encoder: Any


class ProblemTypeDetector:
    """Automatically detect classification vs regression."""
    
    @staticmethod
    def detect(y: np.ndarray, threshold_unique_ratio: float = 0.05) -> str:
        """
        Detect problem type.
        
        Args:
            y: Target values
            threshold_unique_ratio: If unique_ratio < threshold, likely classification
        
        Returns:
            'classification' or 'regression'
        """
        # Check dtype
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            return 'classification'
        
        # Check unique ratio
        unique_ratio = len(np.unique(y)) / len(y)
        if unique_ratio < threshold_unique_ratio:
            return 'classification'
        
        return 'regression'


class SafeLabelEncoder:
    """Safe encoding of target variable."""
    
    def __init__(self):
        self.encoder = None
        self.original_dtype = None
        self.is_encoded = False
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform target variable."""
        self.original_dtype = pd.api.types.infer_dtype(y)
        
        if self.original_dtype in ['string', 'object', 'category']:
            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y.astype(str))
            self.is_encoded = True
            logger.info(f"Encoded target: {len(self.encoder.classes_)} classes")
            return y_encoded
        
        return y
    
    def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
        """Transform back to original format."""
        if self.is_encoded and self.encoder:
            return self.encoder.inverse_transform(y_encoded)
        return y_encoded


class MultiModelTrainer:
    """Train multiple models and select the best."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_encoder = SafeLabelEncoder()
    
    def _get_cv_splitter(self, y: np.ndarray, problem_type: str):
        """Get appropriate cross-validator."""
        if problem_type == 'classification':
            class_counts = np.bincount(np.asarray(y, dtype=int))
            min_class_count = class_counts[class_counts > 0].min() if len(class_counts) else 0
            safe_splits = max(2, min(self.n_splits, int(min_class_count))) if min_class_count else 2
            return StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=self.random_state)
        else:
            safe_splits = max(2, min(self.n_splits, len(y)))
            return KFold(n_splits=safe_splits, shuffle=True, random_state=self.random_state)
    
    def _get_classification_models(self) -> Dict[str, Any]:
        """Get classification models to try."""
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state,
                                                   n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        }
        return models
    
    def _get_regression_models(self) -> Dict[str, Any]:
        """Get regression models to try."""
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state,
                                                 n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        }
        return models
    
    def _get_scoring_metrics(self, problem_type: str) -> Dict[str, str]:
        """Get scoring metrics for cross-validation."""
        if problem_type == 'classification':
            return {
                'accuracy': 'accuracy',
                'f1': 'f1_weighted',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted'
            }
        else:
            return {
                'r2': 'r2',
                'neg_rmse': 'neg_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error'
            }
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              original_features: List[str]) -> ModelingResult:
        """
        Train multiple models and select best.
        
        Args:
            X: Feature array (preprocessed)
            y: Target array
            original_features: Original feature names
        
        Returns:
            ModelingResult object
        """
        # Detect problem type
        problem_type = ProblemTypeDetector.detect(y)
        logger.info(f"Detected problem type: {problem_type}")
        
        # Encode target if needed
        y_processed = self.target_encoder.fit_transform(y)
        
        # Get models
        if problem_type == 'classification':
            models = self._get_classification_models()
        else:
            models = self._get_regression_models()
        
        # Get metrics
        scoring = self._get_scoring_metrics(problem_type)
        cv_splitter = self._get_cv_splitter(y_processed, problem_type)
        
        # Train and evaluate each model
        all_scores = {}
        cross_val_scores = {}
        best_model = None
        best_model_name = None
        best_score = -np.inf
        
        logger.info(f"Training {len(models)} models with {self.n_splits}-fold CV...")
        
        for model_name, model in models.items():
            try:
                # Cross-validation
                cv_results = cross_validate(model, X, y_processed, cv=cv_splitter,
                                           scoring=scoring, n_jobs=-1, return_train_score=True)
                
                # Primary metric (first scoring metric)
                primary_metric = list(scoring.keys())[0]
                test_scores = cv_results[f'test_{primary_metric}']
                mean_cv_score = test_scores.mean()
                
                all_scores[model_name] = mean_cv_score
                cross_val_scores[model_name] = test_scores.tolist()
                
                logger.info(f"  {model_name}: {primary_metric}={mean_cv_score:.4f} "
                           f"(+/- {test_scores.std():.4f})")
                
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_model_name = model_name
                    best_model = model
            
            except Exception as e:
                logger.warning(f"  {model_name}: Failed - {e}")
                continue
        
        if best_model is None:
            raise ValueError("Could not train any models successfully")
        
        # Refit best model on full data
        logger.info(f"Refitting best model: {best_model_name}")
        best_model.fit(X, y_processed)
        
        # Extract feature importance
        feature_importance = self._extract_feature_importance(
            best_model, original_features, X.shape[1]
        )
        
        # Calculate detailed metrics on train/test split
        stratify_target = y_processed if (
            problem_type == 'classification'
            and len(np.unique(y_processed)) > 1
            and np.min(np.bincount(np.asarray(y_processed, dtype=int))) >= 2
        ) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_processed,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify_target,
        )
        
        best_model_final = models[best_model_name].__class__(**models[best_model_name].get_params())
        best_model_final.fit(X_train, y_train)
        
        y_pred = best_model_final.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, problem_type, best_model_final, X_test)
        
        result = ModelingResult(
            problem_type=problem_type,
            best_model_name=best_model_name,
            best_model=best_model,
            best_score=best_score,
            all_scores=all_scores,
            feature_importance=feature_importance,
            original_features=original_features,
            cross_val_scores=cross_val_scores,
            metrics=metrics,
            training_data={
                'n_samples': len(X),
                'n_features': X.shape[1],
                'target_encoded': self.target_encoder.is_encoded
            },
            performance_band=self._categorize_performance(best_score),
            target_encoder=self.target_encoder,
        )
        
        logger.info(f"Modeling complete: best_model={best_model_name}, score={best_score:.4f}")
        return result
    
    def _extract_feature_importance(self, model: Any, feature_names: List[str],
                                   n_features: int, top_n: int = 5) -> Dict[str, float]:
        """Extract feature importance from model."""
        feature_importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_)
                importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                importances = None

            if importances is None:
                logger.debug("Model does not expose feature importance information")
                return feature_importance

            if len(feature_names) != len(importances):
                logger.warning(
                    "Feature count mismatch: %s names vs %s importances. Using generic names.",
                    len(feature_names),
                    len(importances),
                )
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            top_indices = np.argsort(importances)[-top_n:][::-1]
            for idx in top_indices:
                if idx < len(feature_names):
                    feature_importance[feature_names[idx]] = float(importances[idx])
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return feature_importance

    @staticmethod
    def _categorize_performance(score: float) -> str:
        """Map model score to the UI/business-friendly strength bands."""
        if score < 0.6:
            return "weak"
        if score <= 0.75:
            return "moderate"
        return "strong"
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          problem_type: str, model: Any, X_test: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Multi-class or binary?
            unique_classes = len(np.unique(y_true))
            
            if unique_classes == 2:
                # Binary classification: add AUC
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                except Exception as e:
                    logger.debug(f"Could not calculate ROC-AUC: {e}")
            
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        
        else:
            # Regression
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
