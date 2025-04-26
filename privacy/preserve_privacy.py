import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import os

def load_config():
    config_path = Path(__file__).parent.parent / 'config' / 'privacy_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def apply_k_anonymity(df, config):
    k = config['k-anonymity']['k']
    quasi_identifiers = config['k-anonymity']['quasi_identifiers']
    
    # Group by quasi-identifiers and filter groups smaller than k
    groups = df.groupby(quasi_identifiers)
    valid_groups = groups.filter(lambda x: len(x) >= k)
    
    return valid_groups

def apply_l_diversity(df, config):
    l = config['l-diversity']['l']
    sensitive_attribute = config['l-diversity']['sensitive_attribute']
    quasi_identifiers = config['k-anonymity']['quasi_identifiers']
    
    # Group by quasi-identifiers and filter groups with less than l distinct sensitive values
    groups = df.groupby(quasi_identifiers)
    valid_groups = groups.filter(lambda x: x[sensitive_attribute].nunique() >= l)
    
    return valid_groups

def apply_differential_privacy(df, config):
    epsilon = config['differential_privacy']['epsilon']
    numeric_columns = config['differential_privacy']['numeric_columns']
    
    # Add Laplace noise to numeric columns
    for col in numeric_columns:
        if col in df.columns:
            sensitivity = df[col].max() - df[col].min()
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, len(df))
            df[col] = df[col] + noise
    
    return df

def main():
    # Load configuration
    config = load_config()
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
    df = pd.read_csv(data_path)
    
    print("Original data shape:", df.shape)
    print("\nApplying privacy preservation techniques...")
    
    # Apply privacy preservation techniques
    df_k_anon = apply_k_anonymity(df, config)
    print("After k-anonymity shape:", df_k_anon.shape)
    
    df_l_div = apply_l_diversity(df_k_anon, config)
    print("After l-diversity shape:", df_l_div.shape)
    
    df_dp = apply_differential_privacy(df_l_div, config)
    print("After differential privacy shape:", df_dp.shape)
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    output_path = output_dir / 'privacy_preserved_data.csv'
    df_dp.to_csv(output_path, index=False)
    print(f"\nPrivacy preserved data saved to {output_path}")

if __name__ == "__main__":
    main() 