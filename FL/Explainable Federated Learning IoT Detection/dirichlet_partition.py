
import numpy as np
import pandas as pd
from typing import Iterator

# ─────────────────────────────────────────────
# 3. Non-IID Dirichlet partitioning for federated learning
#    Pure-NumPy ClientDataLoader (no torch dependency)
# ─────────────────────────────────────────────

class ClientDataLoader:
    """
    Lightweight DataLoader shim backed by NumPy arrays.
    Provides iterable batches (X_batch, y_batch) compatible with
    standard training loops. Replaces torch.utils.data.DataLoader
    so no heavy framework dependency is required.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self.n_samples = len(X)
        self.n_batches = max(1, int(np.ceil(self.n_samples / batch_size)))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(self.n_samples)
        if self.shuffle:
            self._rng.shuffle(idx)
        for start in range(0, self.n_samples, self.batch_size):
            batch_idx = idx[start: start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __repr__(self) -> str:
        return (
            f"ClientDataLoader(n={self.n_samples}, "
            f"batches={self.n_batches}, batch_size={self.batch_size})"
        )


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    alpha: float,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[list[ClientDataLoader], dict[int, dict]]:
    """
    Partition dataset into K Non-IID clients using Dirichlet distribution.

    Args:
        X         : Feature matrix (N, D), already normalised.
        y         : Integer label array (N,).
        K         : Number of federated clients.
        alpha     : Dirichlet concentration parameter.
                    Low  α (0.1)  → highly heterogeneous (Non-IID).
                    High α (1.0)  → closer to IID.
        batch_size: Batch size for each client's DataLoader.
        seed      : RNG seed for reproducibility.

    Returns:
        client_loaders     : List of K ClientDataLoader objects.
        client_label_dist  : Dict[client_id → Dict[class_id → count]].
    """
    rng = np.random.default_rng(seed=seed)
    n_cls = len(np.unique(y))

    # Group sample indices by class
    class_indices = {c: np.where(y == c)[0] for c in range(n_cls)}

    # Draw Dirichlet proportions: shape (n_cls, K)
    proportions = rng.dirichlet(alpha=[alpha] * K, size=n_cls)

    # Assign samples to clients via Dirichlet shards
    client_index_lists = [[] for _ in range(K)]
    for cls_id, idx in class_indices.items():
        rng.shuffle(idx)
        splits = (np.cumsum(proportions[cls_id]) * len(idx)).astype(int)
        splits = np.clip(splits, 0, len(idx))
        cls_splits = np.split(idx, splits[:-1])          # K sub-arrays
        for k, shard in enumerate(cls_splits):
            client_index_lists[k].extend(shard.tolist())

    # Build loaders and metadata
    client_loaders: list[ClientDataLoader] = []
    client_label_dist: dict[int, dict] = {}

    for k in range(K):
        idx_k = np.array(client_index_lists[k])

        if len(idx_k) == 0:
            # Edge case: empty shard (extreme α with many clients)
            client_loaders.append(
                ClientDataLoader(
                    np.zeros((1, X.shape[1])), np.zeros(1, dtype=int),
                    batch_size=batch_size, seed=k,
                )
            )
            client_label_dist[k] = {}
            continue

        rng.shuffle(idx_k)
        X_k, y_k = X[idx_k], y[idx_k]

        client_loaders.append(
            ClientDataLoader(X_k, y_k, batch_size=batch_size, shuffle=True, seed=k)
        )

        # Label distribution metadata
        _unique, _counts = np.unique(y_k, return_counts=True)
        client_label_dist[k] = {int(c): int(cnt) for c, cnt in zip(_unique, _counts)}

    return client_loaders, client_label_dist


# ─────────────────────────────────────────────
# Run partitioning for all α values & K=10
# ─────────────────────────────────────────────

ALPHA_VALUES = [0.1, 0.5, 1.0]
K_CLIENTS    = 10
BATCH_SIZE   = 64

partition_results: dict[float, dict] = {}

for alpha_val in ALPHA_VALUES:
    loaders, label_dist = partition_data(
        X=X_train, y=y_train,
        K=K_CLIENTS, alpha=alpha_val,
        batch_size=BATCH_SIZE,
    )
    partition_results[alpha_val] = {
        "loaders":    loaders,
        "label_dist": label_dist,
    }

    total_assigned = sum(sum(d.values()) for d in label_dist.values())
    print(f"\n{'='*62}")
    print(f"  α = {alpha_val:<4}  |  K = {K_CLIENTS}  |  Assigned: {total_assigned:,} samples")
    print(f"{'='*62}")
    for k in range(K_CLIENTS):
        dist = label_dist[k]
        n_k  = sum(dist.values())
        top3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{label_map[c]}:{cnt}" for c, cnt in top3)
        print(f"  Client {k:2d}  |  n={n_k:5,d}  |  top-3 → {top_str}")

# Default export: α=0.5
client_loaders            = partition_results[0.5]["loaders"]
client_label_distributions = partition_results[0.5]["label_dist"]

print(f"\n✅ partition_data() validated for α ∈ {ALPHA_VALUES}, K={K_CLIENTS}")
print(f"   'client_loaders' (α=0.5) → {client_loaders[0]}")
print(f"   'client_label_distributions' keys → {list(client_label_distributions.keys())}")
