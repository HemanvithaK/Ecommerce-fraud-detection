from pybloom_live import BloomFilter

bloom = BloomFilter(capacity=1000, error_rate=0.01)
transactions = ['tx001', 'tx002', 'tx003', 'tx001']

for tx in transactions:
    if tx in bloom:
        print(f"⚠️ Possible Duplicate: {tx}")
    else:
        bloom.add(tx)
