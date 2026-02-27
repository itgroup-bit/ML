
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Iterator

# ─────────────────────────────────────────────
# Re-define ClientDataLoader & partition_data locally
# (needed because the class can't cross block boundaries via pickle)
# ─────────────────────────────────────────────

class ClientDataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True, seed=0):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self.n_samples = len(X)
        self.n_batches = max(1, int(np.ceil(self.n_samples / batch_size)))

    def __len__(self): return self.n_batches

    def __iter__(self):
        idx = np.arange(self.n_samples)
        if self.shuffle:
            self._rng.shuffle(idx)
        for start in range(0, self.n_samples, self.batch_size):
            bi = idx[start: start + self.batch_size]
            yield self.X[bi], self.y[bi]

    def __repr__(self):
        return f"ClientDataLoader(n={self.n_samples}, batches={self.n_batches})"


def partition_data(X, y, K, alpha, batch_size=64, seed=42):
    rng = np.random.default_rng(seed=seed)
    n_cls = len(np.unique(y))
    class_indices = {c: np.where(y == c)[0] for c in range(n_cls)}
    proportions = rng.dirichlet(alpha=[alpha] * K, size=n_cls)
    client_index_lists = [[] for _ in range(K)]
    for cls_id, idx in class_indices.items():
        rng.shuffle(idx)
        splits = np.clip((np.cumsum(proportions[cls_id]) * len(idx)).astype(int), 0, len(idx))
        for k, shard in enumerate(np.split(idx, splits[:-1])):
            client_index_lists[k].extend(shard.tolist())
    client_loaders, client_label_dist = [], {}
    for k in range(K):
        idx_k = np.array(client_index_lists[k])
        if len(idx_k) == 0:
            client_loaders.append(ClientDataLoader(
                np.zeros((1, X.shape[1])), np.zeros(1, dtype=int), batch_size=batch_size, seed=k))
            client_label_dist[k] = {}
            continue
        rng.shuffle(idx_k)
        X_k, y_k = X[idx_k], y[idx_k]
        client_loaders.append(ClientDataLoader(X_k, y_k, batch_size=batch_size, shuffle=True, seed=k))
        _u, _c = np.unique(y_k, return_counts=True)
        client_label_dist[k] = {int(c): int(cnt) for c, cnt in zip(_u, _c)}
    return client_loaders, client_label_dist


# ─────────────────────────────────────────────
# Run partitions for visualization
# ─────────────────────────────────────────────

ALPHA_VALUES_VIZ = [0.1, 0.5, 1.0]
K_CLIENTS_VIZ    = 10
BATCH_SIZE_VIZ   = 64

# Store only serializable label_dist dicts (not loaders)
partition_label_dists = {}
partition_loader_reprs = {}

for _alpha in ALPHA_VALUES_VIZ:
    _loaders, _ldist = partition_data(
        X=X_train, y=y_train,
        K=K_CLIENTS_VIZ, alpha=_alpha,
        batch_size=BATCH_SIZE_VIZ,
    )
    partition_label_dists[_alpha] = _ldist
    partition_loader_reprs[_alpha] = [repr(l) for l in _loaders]

# ─────────────────────────────────────────────
# Plot stacked bar charts
# ─────────────────────────────────────────────

BG      = "#1D1D20"
TEXT    = "#fbfbff"
SUBTLE  = "#909094"
PALETTE = [
    "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
    "#1F77B4", "#9467BD", "#8C564B", "#C49C94",
]

classes     = sorted(label_map.keys())
class_names = [label_map[c] for c in classes]

for _alpha in ALPHA_VALUES_VIZ:
    _ldist = partition_label_dists[_alpha]

    mat = np.zeros((K_CLIENTS_VIZ, len(classes)))
    for k in range(K_CLIENTS_VIZ):
        for ci, cls_id in enumerate(classes):
            mat[k, ci] = _ldist[k].get(cls_id, 0)

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_pct = mat / row_sums

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bottoms = np.zeros(K_CLIENTS_VIZ)
    for ci, cls_name in enumerate(class_names):
        ax.bar(range(K_CLIENTS_VIZ), mat_pct[:, ci], bottom=bottoms,
               color=PALETTE[ci % len(PALETTE)], label=cls_name, width=0.72)
        bottoms += mat_pct[:, ci]

    ax.set_xticks(range(K_CLIENTS_VIZ))
    ax.set_xticklabels([f"C{k}" for k in range(K_CLIENTS_VIZ)], color=TEXT, fontsize=11)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=SUBTLE, fontsize=10)
    ax.set_xlim(-0.5, K_CLIENTS_VIZ - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Federated Client", color=TEXT, fontsize=12, labelpad=8)
    ax.set_ylabel("Label Proportion", color=TEXT, fontsize=12, labelpad=8)
    ax.set_title(
        f"Non-IID Dirichlet Partition  |  α = {_alpha}  |  K = {K_CLIENTS_VIZ}",
        color=TEXT, fontsize=14, fontweight="bold", pad=12,
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)

    legend_patches = [
        mpatches.Patch(color=PALETTE[ci % len(PALETTE)], label=n)
        for ci, n in enumerate(class_names)
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, labelcolor=TEXT, fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    print(f"✅ Chart rendered for α = {_alpha}")

print("\nAll partition visualisations complete.")
