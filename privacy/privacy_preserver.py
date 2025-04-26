import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random
from scipy.stats import laplace

class PrivacyPreserver:
    def __init__(self, k: int, l: int, epsilon: float):
        self.k = k  # k-anonymity parameter
        self.l = l  # l-diversity parameter
        self.epsilon = epsilon  # differential privacy parameter

    def apply_k_anonymity(self, data: Dict[str, Any], quasi_identifiers: List[str]) -> Dict[str, Any]:
        """Apply k-anonymity by generalizing quasi-identifiers."""
        anonymized_data = data.copy()
        
        for identifier in quasi_identifiers:
            if identifier in data:
                if identifier == 'location':
                    # Generalize location to region
                    anonymized_data[identifier] = f"Region_{data[identifier][:2]}"
                elif identifier == 'payment_method':
                    # Generalize payment method to category
                    anonymized_data[identifier] = "Payment"
                elif identifier == 'customer_id':
                    # Generalize customer ID to group
                    anonymized_data[identifier] = f"Group_{data[identifier][:2]}"
        
        return anonymized_data

    def apply_l_diversity(self, data: Dict[str, Any], sensitive_attributes: List[str]) -> Dict[str, Any]:
        """Apply l-diversity by ensuring diversity in sensitive attributes."""
        diversified_data = data.copy()
        
        for attribute in sensitive_attributes:
            if attribute in data:
                if attribute == 'amount':
                    # Add noise to amount
                    noise = np.random.laplace(0, 1/self.epsilon)
                    diversified_data[attribute] = round(data[attribute] + noise, 2)
                elif attribute == 'is_fraud':
                    # Ensure diversity in fraud status
                    if random.random() < 0.1:  # 10% chance to flip
                        diversified_data[attribute] = not data[attribute]
        
        return diversified_data

    def apply_differential_privacy(self, data: Dict[str, Any], sensitive_attributes: List[str]) -> Dict[str, Any]:
        """Apply differential privacy by adding Laplace noise to sensitive attributes."""
        private_data = data.copy()
        
        for attribute in sensitive_attributes:
            if attribute in data:
                if isinstance(data[attribute], (int, float)):
                    # Add Laplace noise for numerical values
                    noise = np.random.laplace(0, 1/self.epsilon)
                    private_data[attribute] = round(data[attribute] + noise, 2)
                elif isinstance(data[attribute], bool):
                    # Apply randomized response for boolean values
                    if random.random() < 1/(1 + np.exp(self.epsilon)):
                        private_data[attribute] = not data[attribute]
        
        return private_data

    def apply_privacy_techniques(self, data: Dict[str, Any], 
                               quasi_identifiers: List[str],
                               sensitive_attributes: List[str]) -> Dict[str, Any]:
        """Apply all privacy techniques to the data."""
        # Apply k-anonymity
        anonymized_data = self.apply_k_anonymity(data, quasi_identifiers)
        
        # Apply l-diversity
        diversified_data = self.apply_l_diversity(anonymized_data, sensitive_attributes)
        
        # Apply differential privacy
        private_data = self.apply_differential_privacy(diversified_data, sensitive_attributes)
        
        return private_data

    def k_anonymize(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """
        Implement k-anonymity by generalizing quasi-identifiers
        
        Args:
            df: Input DataFrame
            quasi_identifiers: List of columns to be generalized
            
        Returns:
            DataFrame with k-anonymized data
        """
        # Create a copy to avoid modifying original data
        anonymized_df = df.copy()
        
        # Group by quasi-identifiers and ensure each group has at least k records
        groups = anonymized_df.groupby(quasi_identifiers)
        
        # Generalize values in each group
        for _, group in groups:
            if len(group) < self.k:
                # If group is too small, generalize further
                for col in quasi_identifiers:
                    if pd.api.types.is_numeric_dtype(anonymized_df[col]):
                        # For numeric columns, use range
                        min_val = group[col].min()
                        max_val = group[col].max()
                        anonymized_df.loc[group.index, col] = f"[{min_val}-{max_val}]"
                    else:
                        # For categorical columns, use first few characters
                        anonymized_df.loc[group.index, col] = group[col].str[:3] + "***"
        
        return anonymized_df

    def l_diversity(self, df: pd.DataFrame, quasi_identifiers: List[str], 
                   sensitive_attributes: List[str]) -> pd.DataFrame:
        """
        Implement l-diversity by ensuring each equivalence class has at least l distinct
        sensitive values
        
        Args:
            df: Input DataFrame
            quasi_identifiers: List of columns used for grouping
            sensitive_attributes: List of sensitive columns to protect
            
        Returns:
            DataFrame with l-diverse data
        """
        # Create a copy to avoid modifying original data
        diverse_df = df.copy()
        
        # Group by quasi-identifiers
        groups = diverse_df.groupby(quasi_identifiers)
        
        for _, group in groups:
            for sensitive_col in sensitive_attributes:
                unique_values = group[sensitive_col].nunique()
                if unique_values < self.l:
                    # If not enough diversity, add noise
                    diverse_df.loc[group.index, sensitive_col] = self._add_noise(
                        group[sensitive_col], 
                        self.epsilon
                    )
        
        return diverse_df

    def differential_privacy(self, df: pd.DataFrame, 
                           sensitive_columns: List[str]) -> pd.DataFrame:
        """
        Implement differential privacy by adding Laplace noise to sensitive columns
        
        Args:
            df: Input DataFrame
            sensitive_columns: List of columns to apply differential privacy to
            
        Returns:
            DataFrame with differentially private data
        """
        # Create a copy to avoid modifying original data
        private_df = df.copy()
        
        for col in sensitive_columns:
            if pd.api.types.is_numeric_dtype(private_df[col]):
                # Add Laplace noise to numeric columns
                private_df[col] = self._add_noise(private_df[col], self.epsilon)
            else:
                # For categorical columns, use randomized response
                private_df[col] = self._randomized_response(private_df[col], self.epsilon)
        
        return private_df

    def _add_noise(self, series: pd.Series, epsilon: float) -> pd.Series:
        """Add Laplace noise to a numeric series"""
        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale, len(series))
        return series + noise

    def _randomized_response(self, series: pd.Series, epsilon: float) -> pd.Series:
        """Implement randomized response for categorical data"""
        p = 1.0 / (1.0 + np.exp(epsilon))
        mask = np.random.random(len(series)) < p
        randomized = series.copy()
        randomized[mask] = random.choice(series.unique())
        return randomized

    def apply_privacy_techniques(self, df: pd.DataFrame, 
                               quasi_identifiers: List[str],
                               sensitive_attributes: List[str]) -> pd.DataFrame:
        """
        Apply all privacy techniques in sequence
        
        Args:
            df: Input DataFrame
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive columns
            
        Returns:
            DataFrame with all privacy techniques applied
        """
        # First apply k-anonymity
        anonymized = self.k_anonymize(df, quasi_identifiers)
        
        # Then apply l-diversity
        diverse = self.l_diversity(anonymized, quasi_identifiers, sensitive_attributes)
        
        # Finally apply differential privacy
        private = self.differential_privacy(diverse, sensitive_attributes)
        
        return private 