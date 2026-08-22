"""Step 3 — Build the embedding index for the historical repository.

Always re-encodes: running this script means you want a fresh index, typically
because the repository was regenerated. Retrieval works without it — get_index()
builds the index on first use — but running it explicitly keeps the two-minute
encoding out of a demo, and out of the first experiment run.

The first run downloads the encoder (~90 MB) and needs an internet connection.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
from agents.retrieval_agent import get_index, retrieve

repo = pd.read_csv(config.HISTORICAL_CSV)
embeddings = get_index(repo, rebuild=True)

print(f"\nindex: {embeddings.shape} -> {config.EMBEDDINGS_FILE}")

# A description from the repository must retrieve itself, with similarity ~1.
# Cheapest possible check that the encoding and the row order are correct.
hit = retrieve(repo.description.iloc[0], repo, embeddings, k=1)
print(f"self-retrieval similarity: {hit.similarity.iloc[0]:.4f} (expected ~1.0)")