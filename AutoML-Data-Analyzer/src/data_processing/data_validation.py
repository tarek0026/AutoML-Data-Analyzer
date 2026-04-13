"""
Data Validation & Feature Detection Module.

Handles:
- Automatic detection of ID-like columns
- Detection of constant/near-constant features
- Removal of problematic features
- Target column validation
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
import logging
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)


class DataValidator:
    """Production-grade data validation and feature filtering."""
    
    # Keywords indicating ID-like columns
    ID_KEYWORDS = ["id", "index", "idx", "key", "pk", "identifier", "code", 
                   "record_id", "row_id", "_id", "uid", "uuid", "serial"]
    
    # Keywords indicating date/time columns
    DATE_KEYWORDS = ["date", "time", "timestamp", "created", "updated", "datetime"]
    
    # Keywords indicating potentially useless columns
    USELESS_KEYWORDS = ["name", "description", "notes", "comment", "text", "remarks"]
    
    def __init__(self, id_threshold: float = 0.95, constant_threshold: float = 0.01):
        """
        Initialize validator.
        
        Args:
            id_threshold: If unique_ratio > threshold, likely ID column
            constant_threshold: Coefficient of variation threshold for constant detection
        """
        self.id_threshold = id_threshold
        self.constant_threshold = constant_threshold
        self.removed_features = {}
    
    def is_id_like_column(self, col_name: str) -> bool:
        """Check if column name suggests it's an identifier."""
        col_lower = col_name.lower().strip()
        return any(keyword in col_lower for keyword in self.ID_KEYWORDS)
    
    def is_date_column(self, col_name: str) -> bool:
        """Check if column name suggests it's a date/time."""
        col_lower = col_name.lower().strip()
        return any(keyword in col_lower for keyword in self.DATE_KEYWORDS)
    
    def is_text_column(self, col_name: str) -> bool:
        """Check if column is primarily text (not useful for ML)."""
        col_lower = col_name.lower().strip()
        return any(keyword in col_lower for keyword in self.USELESS_KEYWORDS)
    
    def is_high_cardinality(self, series: pd.Series, threshold: float = None) -> bool:
        """
        Check if column has too many unique values (likely ID).
        
        Args:
            series: Column data
            threshold: Ratio of unique values. If None, uses self.id_threshold
        
        Returns:
            True if unique_ratio > threshold
        """
        if threshold is None:
            threshold = self.id_threshold
        
        if len(series) == 0:
            return False
        
        unique_ratio = len(series.unique()) / len(series)
        return unique_ratio > threshold
    
    def is_constant_or_near_constant(self, series: pd.Series) -> bool:
        """
        Check if feature has near-zero variance.
        
        Args:
            series: Column data
        
        Returns:
            True if coefficient of variation < threshold
        """
        if len(series) < 2:
            return True
        
        # Skip NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < 2:
            return True
        
        # For non-numeric data, avoid numeric reductions such as std/mean.
        if not is_numeric_dtype(series_clean):
            unique_ratio = len(series_clean.unique()) / len(series_clean)
            return unique_ratio < 0.01
        
        # For numeric, check coefficient of variation
        std = series_clean.std()
        mean = series_clean.mean()
        
        if abs(mean) < 1e-10:
            return std < 1e-10
        
        cv = abs(std / mean)
        return cv < self.constant_threshold
    
    def detect_problematic_features(self, df: pd.DataFrame, target_col: str = None) -> Tuple[List[str], dict]:
        """
        Detect all problematic features that should be removed.
        
        Args:
            df: Input dataframe
            target_col: Name of target column (will be excluded from removal)
        
        Returns:
            Tuple of (list of columns to remove, dict with removal reasons)
        """
        columns_to_remove = []
        reasons = {}
        
        for col in df.columns:
            # Never remove target column
            if target_col and col == target_col:
                continue
            
            reason = None
            
            # Check for ID-like names
            if self.is_id_like_column(col):
                reason = "ID-like column name"
            
            # Check for date columns
            elif self.is_date_column(col):
                reason = "Date/Time column (not useful for ML)"
            
            # Check for text columns
            elif self.is_text_column(col):
                reason = "Text/Description column (high cardinality)"
            
            # Check for high cardinality in numeric/categorical
            elif self.is_high_cardinality(df[col]):
                reason = f"High cardinality ({len(df[col].unique())} unique values, likely ID)"
            
            # Check for constant/near-constant features
            elif self.is_constant_or_near_constant(df[col]):
                reason = "Near-constant feature (no variance)"
            
            if reason:
                columns_to_remove.append(col)
                reasons[col] = reason
        
        self.removed_features = reasons
        return columns_to_remove, reasons
    
    def validate_and_clean(self, df: pd.DataFrame, target_col: str = None, 
                           verbose: bool = True) -> pd.DataFrame:
        """
        Automatically detect and remove problematic features.
        
        Args:
            df: Input dataframe
            target_col: Target column name (won't be removed)
            verbose: Whether to log removed features
        
        Returns:
            Cleaned dataframe
        """
        cols_to_remove, reasons = self.detect_problematic_features(df, target_col)
        
        if verbose and cols_to_remove:
            logger.info(f"Removing {len(cols_to_remove)} problematic features:")
            for col, reason in reasons.items():
                logger.info(f"  - {col}: {reason}")
        
        df_clean = df.drop(columns=cols_to_remove)
        
        return df_clean
    
    def get_removed_features_report(self) -> dict:
        """Get report of all removed features and reasons."""
        return self.removed_features.copy()


def detect_feature_leakage(X_features: pd.DataFrame) -> List[str]:
    """
    Utility function to detect commonly problematic features.
    
    Args:
        X_features: Feature dataframe
    
    Returns:
        List of columns that should be excluded
    """
    validator = DataValidator()
    columns_to_remove, _ = validator.detect_problematic_features(X_features)
    return columns_to_remove
