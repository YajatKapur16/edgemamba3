# scripts/download_data.py

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.datasets import LRGBDataset
from relbench.datasets import get_dataset
from relbench.tasks import get_task

print("Downloading Peptides-func...")
LRGBDataset(root="./data/cache/lrgb", name="Peptides-func")

print("Downloading Peptides-struct...")
LRGBDataset(root="./data/cache/lrgb", name="Peptides-struct")

print("Downloading rel-hm...")
get_dataset("rel-hm", download=True)
get_task("rel-hm", "user-churn", download=True)

print("Downloading rel-amazon...")
get_dataset("rel-amazon", download=True)
get_task("rel-amazon", "user-ltv", download=True)

print("All data downloaded.")
