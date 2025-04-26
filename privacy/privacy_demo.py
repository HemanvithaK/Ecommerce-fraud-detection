import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def load_config():
    """Load privacy configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'config' / 'privacy_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """Load sample transaction data."""
    data_path = Path(__file__).parent / 'sample_data.csv'
    return pd.read_csv(data_path)

def apply_k_anonymity(df, k, quasi_identifiers):
    """Apply k-anonymity to the dataset."""
    # Group by quasi-identifiers and ensure each group has at least k records
    groups = df.groupby(quasi_identifiers)
    return df[groups.transform('size') >= k]

def apply_l_diversity(df, l, sensitive_attribute, quasi_identifiers):
    """Apply l-diversity to the dataset."""
    # Group by quasi-identifiers and ensure each group has at least l distinct values
    groups = df.groupby(quasi_identifiers)
    return df[groups[sensitive_attribute].transform('nunique') >= l]

def apply_differential_privacy(df, epsilon, columns):
    """Apply differential privacy to numeric columns."""
    # Add Laplace noise to numeric columns
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            scale = 1.0 / epsilon
            noise = np.random.laplace(0, scale, len(df))
            df[col] = df[col] + noise
    return df

def main():
    # Load configuration and data
    config = load_config()
    df = load_data()
    
    print("Original dataset:")
    print(df)
    print("\nShape:", df.shape)
    
    # Apply k-anonymity
    k = config['k_anonymity']['k']
    quasi_identifiers = config['k_anonymity']['quasi_identifiers']
    df_k_anon = apply_k_anonymity(df, k, quasi_identifiers)
    
    print("\nAfter k-anonymity (k={}):".format(k))
    print(df_k_anon)
    print("Shape:", df_k_anon.shape)
    
    # Apply l-diversity
    l = config['l_diversity']['l']
    sensitive_attribute = config['l_diversity']['sensitive_attribute']
    df_l_div = apply_l_diversity(df_k_anon, l, sensitive_attribute, quasi_identifiers)
    
    print("\nAfter l-diversity (l={}):".format(l))
    print(df_l_div)
    print("Shape:", df_l_div.shape)
    
    # Apply differential privacy
    epsilon = config['differential_privacy']['epsilon']
    numeric_columns = config['differential_privacy']['numeric_columns']
    df_dp = apply_differential_privacy(df_l_div, epsilon, numeric_columns)
    
    print("\nAfter differential privacy (epsilon={}):".format(epsilon))
    print(df_dp)
    print("Shape:", df_dp.shape)
    
    # Save protected data
    output_path = Path(__file__).parent.parent.parent / 'output' / 'protected_data.csv'
    df_dp.to_csv(output_path, index=False)
    print("\nProtected data saved to:", output_path)

if __name__ == "__main__":
    main() 