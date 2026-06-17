from __future__ import annotations

import argparse
import random
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from data.simulation.init_embeddings import initialize_all_clip_embeddings
from data.simulation.utils import CLIP_PARAMETERS
from utils.naming import clip_embedding_name

NUM_PARAMETERS = 10
EMBEDDING_DIM = 512


def sample_combined_clip_vector(clip_embeddings: dict, rng: random.Random) -> np.ndarray:
    combined = np.zeros(NUM_PARAMETERS * EMBEDDING_DIM, dtype=np.float32)
    for index, (_, parameter_enum) in enumerate(CLIP_PARAMETERS[:NUM_PARAMETERS]):
        chosen = rng.choice(list(parameter_enum))
        embedding = clip_embeddings[parameter_enum.__name__][clip_embedding_name(chosen)]
        combined[index * EMBEDDING_DIM:(index + 1) * EMBEDDING_DIM] = embedding
    return combined


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def collect_similarities(
    clip_embeddings: dict, num_runs: int, seed: Optional[int] = None
) -> List[float]:
    rng = random.Random(seed)
    return [
        cosine_similarity(
            sample_combined_clip_vector(clip_embeddings, rng),
            sample_combined_clip_vector(clip_embeddings, rng),
        )
        for _ in range(num_runs)
    ]


def plot_histogram(similarities: List[float], num_bins: int, out_path: str) -> None:
    plt.figure(figsize=(12, 8))
    plt.hist(similarities, bins=num_bins)
    plt.xlabel("Cosine similarity")
    plt.title("Cosine similarity histogram")
    plt.xlim([-1, 1])
    plt.grid()
    plt.savefig(out_path)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="cosine_similarities_hist.jpeg")
    args = parser.parse_args()

    clip_embeddings = initialize_all_clip_embeddings()
    similarities = collect_similarities(clip_embeddings, args.runs, args.seed)
    plot_histogram(similarities, args.bins, args.out)


if __name__ == "__main__":
    main()
