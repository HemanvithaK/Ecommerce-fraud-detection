import json
import random
from datetime import datetime, timedelta
import csv
import os

def generate_transaction():
    """Generate a sample e-commerce transaction"""
    return {
        'transaction_id': f'TX{random.randint(10000, 99999)}',
        'user_id': f'USER{random.randint(1, 1000)}',
        'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        'amount': round(random.uniform(10, 1000), 2),
        'product_id': f'PROD{random.randint(1, 100)}',
        'category': random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Food'])
    }

def generate_sample_data(num_records=1000):
    """Generate sample transaction data and save to CSV"""
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Generate transactions
    transactions = [generate_transaction() for _ in range(num_records)]
    
    # Save to CSV
    output_file = '../data/transactions.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=transactions[0].keys())
        writer.writeheader()
        writer.writerows(transactions)
    
    print(f"Generated {num_records} transactions and saved to {output_file}")
    return output_file

if __name__ == "__main__":
    generate_sample_data()
