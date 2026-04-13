"""
Production-grade preprocessing pipeline.

This module now exposes a single canonical preprocessing flow:

raw dataframe
-> remove ID-like columns
-> handle missing values / encode categoricals / scale numerics
-> optional PCA for visualization only

The orchestration layer uses this module exclusively so downstream
clustering and modeling never mix raw and processed data.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class RobustPreprocessor:
    """
    Production-grade preprocessing with automatic feature type detection
    and proper handling of missing values.
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 use_knn_imputer: bool = False,
                 knn_neighbors: int = 5,
                 scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            numeric_strategy: 'mean', 'median', 'knn'
            categorical_strategy: 'most_frequent', 'constant'
            use_knn_imputer: Whether to use KNNImputer for numeric features
            knn_neighbors: Number of neighbors for KNN imputation
            scaler_type: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors
        self.scaler_type = scaler_type
        
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
        self.fitted = False
    
    def _detect_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Detect numeric and categorical features.
        
        Args:
            X: Input dataframe
        
        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return numeric, categorical
    
    def _build_pipeline(self) -> ColumnTransformer:
        """Build sklearn ColumnTransformer pipeline."""
        
        transformers = []
        
        # Numeric preprocessing
        if self.numeric_features and len(self.numeric_features) > 0:
            if self.use_knn_imputer:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', KNNImputer(n_neighbors=self.knn_neighbors)),
                    ('scaler', StandardScaler() if self.scaler_type == 'standard' 
                     else MinMaxScaler())
                ])
            else:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=self.numeric_strategy)),
                    ('scaler', StandardScaler() if self.scaler_type == 'standard' 
                     else MinMaxScaler())
                ])
            
            transformers.append(('num', numeric_transformer, self.numeric_features))
        
        # Categorical preprocessing
        if self.categorical_features and len(self.categorical_features) > 0:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.categorical_strategy, 
                                         fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            transformers.append(('cat', categorical_transformer, self.categorical_features))
        
        if not transformers:
            raise ValueError("No features to preprocess")
        
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def fit(self, X: pd.DataFrame) -> 'RobustPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training dataframe
        
        Returns:
            Self
        """
        # Detect feature types
        self.numeric_features, self.categorical_features = self._detect_feature_types(X)
        
        logger.info(f"Detected {len(self.numeric_features)} numeric features: {self.numeric_features}")
        logger.info(f"Detected {len(self.categorical_features)} categorical features: {self.categorical_features}")
        
        # Build and fit pipeline
        self.preprocessor = self._build_pipeline()
        self.preprocessor.fit(X)
        
        self.fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform
        
        Returns:
            Transformed numpy array
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Data to fit and transform
        
        Returns:
            Transformed numpy array
        """
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names_out(self) -> np.ndarray:
        """
        Get output feature names after transformation.
        
        Returns:
            Array of feature names
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return self.preprocessor.get_feature_names_out()
    
    def get_preprocessing_summary(self) -> dict:
        """Get summary of preprocessing configuration."""
        return {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'numeric_strategy': self.numeric_strategy,
            'categorical_strategy': self.categorical_strategy,
            'use_knn_imputer': self.use_knn_imputer,
            'scaler_type': self.scaler_type,
            'fitted': self.fitted
        }


def _detect_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect ID-like columns to be excluded.
    
    Args:
        df: Input dataframe
    
    Returns:
        List of column names that appear to be IDs
    """
    id_patterns = ['id', 'index', 'uid', 'unique_id', 'row_id', 'pk', 'primary_key']
    id_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name matches ID patterns (exact word match or contains)
        if col_lower == 'id' or col_lower.endswith('_id') or any(f'_{p}' in col_lower or f'{p}_' in col_lower for p in ['id', 'index', 'uid']):
            id_columns.append(col)
    
    return id_columns


def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize object columns before sklearn preprocessing."""
    coerced = df.copy()
    object_cols = coerced.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        coerced[col] = coerced[col].replace(r"^\s*$", np.nan, regex=True)

    return coerced


def preprocess_pipeline(
    df: pd.DataFrame,
    use_pca: bool = False,
    pca_components: int = 2,
    exclude_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    use_knn: bool = False,
) -> pd.DataFrame:
    """
    Canonical preprocessing entrypoint for the application.

    This returns a numeric, model-ready dataframe and is the only dataframe
    that should be used for clustering/modeling after preprocessing begins.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is None or empty")

    working_df = _coerce_object_columns(df.copy())

    # Remove duplicated records early to keep all downstream arrays aligned.
    working_df = working_df.drop_duplicates().copy()

    cols_to_drop = set(exclude_cols or [])
    cols_to_drop.update(_detect_id_columns(working_df))
    if target_col and target_col in working_df.columns:
        cols_to_drop.add(target_col)

    valid_drop_cols = [col for col in cols_to_drop if col in working_df.columns]
    if valid_drop_cols:
        working_df = working_df.drop(columns=valid_drop_cols)

    if working_df.empty or working_df.shape[1] == 0:
        raise ValueError("No usable features remain after preprocessing")

    preprocessor = RobustPreprocessor(use_knn_imputer=use_knn)
    transformed = preprocessor.fit_transform(working_df)
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out().tolist()
    ]

    processed_df = pd.DataFrame(transformed, columns=feature_names, index=working_df.index)

    if use_pca:
        return apply_pca(processed_df, n_components=pca_components)

    return processed_df


def apply_pca(df: pd.DataFrame, n_components: int = 2) -> Optional[pd.DataFrame]:
    """
    Apply PCA safely to numeric data only for visualization use-cases.

    Returns None when PCA cannot be computed safely.
    """
    if df is None or df.empty:
        return None

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    if numeric_df.empty or numeric_df.shape[0] < 3 or numeric_df.shape[1] < 2:
        logger.warning("PCA skipped: insufficient numeric rows or columns")
        return None

    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    safe_components = min(n_components, numeric_df.shape[0], numeric_df.shape[1])
    if safe_components < 2:
        logger.warning("PCA skipped: fewer than 2 safe components available")
        return None

    try:
        pca = PCA(n_components=safe_components)
        transformed = pca.fit_transform(numeric_df)
        return pd.DataFrame(
            transformed,
            columns=[f"PC{i + 1}" for i in range(safe_components)],
            index=numeric_df.index,
        )
    except Exception as exc:
        logger.warning("PCA failed safely: %s", exc)
        return None


def preprocess_for_clustering(df: pd.DataFrame, 
                               exclude_cols: List[str] = None,
                               target_col: str = None) -> Tuple[np.ndarray, pd.DataFrame, RobustPreprocessor]:
    """
    Prepare data for clustering with proper scaling and feature selection.
    
    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude (IDs, dates, etc.)
        target_col: Target column to exclude
    
    Returns:
        Tuple of (preprocessed_array, processed_dataframe, preprocessor_object)
    """
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is None or empty")
    
    df_processed = preprocess_pipeline(
        df,
        use_pca=False,
        exclude_cols=exclude_cols,
        target_col=target_col,
        use_knn=False,
    )
    X_processed = df_processed.to_numpy()

    # Refit a preprocessor on the aligned, pre-dropped feature set for metadata access.
    base_df = _coerce_object_columns(df.copy()).drop_duplicates().copy()
    cols_to_drop = set(exclude_cols or [])
    cols_to_drop.update(_detect_id_columns(base_df))
    if target_col and target_col in base_df.columns:
        cols_to_drop.add(target_col)
    aligned_df = base_df.drop(columns=[c for c in cols_to_drop if c in base_df.columns])
    preprocessor = RobustPreprocessor(use_knn_imputer=False).fit(aligned_df)
    
    logger.info(f"Clustering preprocessing: {aligned_df.shape[0]} samples, {aligned_df.shape[1]} features → "
               f"{X_processed.shape[1]} processed features")
    
    return X_processed, df_processed, preprocessor


def preprocess_for_modeling(df: pd.DataFrame,
                             target_col: str,
                             exclude_cols: List[str] = None,
                             use_knn: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str], RobustPreprocessor]:
    """
    Prepare data for supervised learning (classification/regression).
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        exclude_cols: Columns to exclude
        use_knn: Whether to use KNN imputation
    
    Returns:
        Tuple of (X_preprocessed, y, df_processed, feature_names, preprocessor)
    """
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is None or empty")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    working_df = _coerce_object_columns(df.copy()).drop_duplicates().copy()
    y = working_df[target_col].copy()

    df_processed = preprocess_pipeline(
        working_df,
        use_pca=False,
        exclude_cols=exclude_cols,
        target_col=target_col,
        use_knn=use_knn,
    )
    X_processed = df_processed.to_numpy()
    original_features = df_processed.columns.tolist()

    feature_df = working_df.drop(columns=[target_col])
    cols_to_drop = set(exclude_cols or [])
    cols_to_drop.update(_detect_id_columns(feature_df))
    aligned_features = feature_df.drop(columns=[c for c in cols_to_drop if c in feature_df.columns])
    preprocessor = RobustPreprocessor(use_knn_imputer=use_knn).fit(aligned_features)
    
    logger.info(f"Modeling preprocessing: {aligned_features.shape[0]} samples, {aligned_features.shape[1]} features → "
               f"{X_processed.shape[1]} processed features")
    
    return X_processed, y.values, df_processed, original_features, preprocessor
