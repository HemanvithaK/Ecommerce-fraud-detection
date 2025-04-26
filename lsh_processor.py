import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import json
from typing import List, Dict, Any
import os

class SimilarityProcessor:
    def __init__(self, n_neighbors: int = 5, radius: float = 1.0):
        """
        Initialize similarity processor
        Args:
            n_neighbors: Number of neighbors to find
            radius: Radius for considering points as similar
        """
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm='ball_tree')
        self.scaler = StandardScaler()
        self.transaction_ids = []

    def _create_feature_vector(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Convert transaction data into a feature vector
        """
        # Convert categorical features to numerical
        payment_method_map = {'credit_card': 0, 'debit_card': 1, 'paypal': 2}
        location_map = {'US': 0, 'UK': 1, 'CA': 2, 'DE': 3, 'AU': 4, 'FR': 5, 'CN': 6, 'BR': 7, 'RU': 8}
        
        features = [
            float(transaction['amount']),
            payment_method_map.get(transaction['payment_method'], -1),
            location_map.get(transaction['location'], -1)
        ]
        return np.array(features).reshape(1, -1)

    def process_batch(self, transactions: List[Dict[str, Any]], output_file: str):
        """
        Process a batch of transactions and find similar groups
        """
        # Extract features and IDs
        features = []
        self.transaction_ids = []
        for transaction in transactions:
            features.append(self._create_feature_vector(transaction)[0])
            self.transaction_ids.append(transaction['transaction_id'])
        
        # Convert to numpy array and scale features
        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit nearest neighbors model
        self.nn.fit(X_scaled)
        
        # Find similar transactions
        similar_groups = {}
        for i, transaction in enumerate(transactions):
            # Get nearest neighbors
            distances, indices = self.nn.kneighbors(X_scaled[i:i+1])
            
            # Only include groups with more than one transaction and within radius
            if len(indices[0]) > 1 and distances[0][1] < self.radius:
                similar_ids = [self.transaction_ids[idx] for idx in indices[0]]
                similar_groups[transaction['transaction_id']] = {
                    'similar_transactions': similar_ids,
                    'features': features[i].tolist(),
                    'distances': distances[0].tolist()
                }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(similar_groups, f, indent=2)
        
        print(f"Processed {len(transactions)} transactions")
        print(f"Found {len(similar_groups)} groups of similar transactions")
        
        # Print some example groups
        if similar_groups:
            print("\nExample similar transaction groups:")
            for i, (tid, group) in enumerate(list(similar_groups.items())[:3]):
                print(f"\nGroup {i+1}:")
                print(f"Transaction ID: {tid}")
                print(f"Similar transactions: {len(group['similar_transactions'])}")
                print(f"Distances: {[round(d, 3) for d in group['distances']]}")
        
        return similar_groups

def main():
    # Example usage
    input_file = "data/transactions.json"
    output_file = "output/similar_transactions.json"
    
    # Read transactions
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            transactions = [json.loads(line) for line in f]
    else:
        print(f"Input file {input_file} not found")
        return

    # Process transactions
    processor = SimilarityProcessor(n_neighbors=5, radius=1.0)
    similar_groups = processor.process_batch(transactions, output_file)
    
    # Print summary
    print(f"\nResults saved to {output_file}")
    print(f"Total groups of similar transactions: {len(similar_groups)}")

if __name__ == "__main__":
    main() 