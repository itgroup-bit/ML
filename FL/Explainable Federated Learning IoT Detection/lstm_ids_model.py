
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
#  LSTM-Based IDS Model — NumPy Implementation
#
#  Architecture (mirrors a PyTorch LSTM IDS model):
#    Input  (batch, seq_len, input_dim)
#      └─→  Input projection  : Linear(input_dim → hidden_dim) + LayerNorm + ReLU
#      └─→  Bidirectional LSTM: 2-layer stacked BiLSTM (manual numpy cells)
#      └─→  Classifier head   : Linear(hidden*2 → hidden) + ReLU + Linear(hidden → num_classes)
#
#  Training:
#    Loss      : Cross-Entropy
#    Optimizer : Adam (per-parameter m/v moments)
#    FedProx   : proximal_loss = (μ/2) * Σ ‖w_local − w_global‖²
# ─────────────────────────────────────────────────────────────────────────────

class LSTMCell:
    """Single LSTM cell (manual numpy forward + gradient storage)."""

    def __init__(self, input_dim: int, hidden_dim: int, rng):
        k = 1.0 / hidden_dim ** 0.5
        # Combined (forget, input, gate, output) weight matrices
        self.Wh = rng.uniform(-k, k, (4 * hidden_dim, hidden_dim))
        self.Wx = rng.uniform(-k, k, (4 * hidden_dim, input_dim))
        self.b  = np.zeros(4 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        """x: (hidden_dim,), returns h_next, c_next"""
        gates = self.Wx @ x + self.Wh @ h_prev + self.b
        hd = self.hidden_dim
        f = _sigmoid(gates[:hd])
        i = _sigmoid(gates[hd:2*hd])
        g = np.tanh(gates[2*hd:3*hd])
        o = _sigmoid(gates[3*hd:])
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        return h_next, c_next

    def params(self):
        return [self.Wh, self.Wx, self.b]


class LSTMIdsModel:
    """
    Bidirectional 2-layer LSTM IDS for multi-class traffic classification.

    Supports FedProx via ``proximal_loss(global_params, mu)``.
    Trained with ``train_step(X_batch, y_batch, lr, global_params, mu)``.
    """

    def __init__(
        self,
        input_dim:    int   = 78,
        hidden_dim:   int   = 128,
        num_classes:  int   = 5,
        num_layers:   int   = 2,
        dropout:      float = 0.3,
        seed:         int   = 0,
    ):
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.dropout     = dropout
        rng = np.random.default_rng(seed)

        # ── Input projection: input_dim → hidden_dim ─────────────────────
        self.W_proj = rng.uniform(-0.1, 0.1, (hidden_dim, input_dim))
        self.b_proj = np.zeros(hidden_dim)
        self.ln_gamma = np.ones(hidden_dim)
        self.ln_beta  = np.zeros(hidden_dim)

        # ── Stacked BiLSTM (forward + backward, 2 layers) ────────────────
        # Layer 1: input is projected (hidden_dim) on each direction
        self.fwd_cells = [LSTMCell(hidden_dim, hidden_dim, rng) for _ in range(num_layers)]
        self.bwd_cells = [LSTMCell(hidden_dim, hidden_dim, rng) for _ in range(num_layers)]

        # ── Classifier head: hidden*2 → hidden → num_classes ─────────────
        lstm_out = hidden_dim * 2
        self.W_fc1 = rng.uniform(-0.1, 0.1, (hidden_dim, lstm_out))
        self.b_fc1 = np.zeros(hidden_dim)
        self.W_fc2 = rng.uniform(-0.1, 0.1, (num_classes, hidden_dim))
        self.b_fc2 = np.zeros(num_classes)

        # ── Adam optimizer state ──────────────────────────────────────────
        self._init_adam()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _all_params(self):
        """Flat list of all weight arrays."""
        p = [self.W_proj, self.b_proj, self.ln_gamma, self.ln_beta]
        for cell in self.fwd_cells + self.bwd_cells:
            p.extend(cell.params())
        p.extend([self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2])
        return p

    def _init_adam(self):
        self._adam_m = [np.zeros_like(p) for p in self._all_params()]
        self._adam_v = [np.zeros_like(p) for p in self._all_params()]
        self._adam_t = 0

    def _project(self, x_seq):
        """Input projection + LayerNorm + ReLU per time-step."""
        # x_seq: (T, input_dim) → out: (T, hidden_dim)
        h = x_seq @ self.W_proj.T + self.b_proj          # (T, hidden)
        # LayerNorm per step
        mu_ln = h.mean(axis=-1, keepdims=True)
        std_ln = h.std(axis=-1, keepdims=True) + 1e-8
        h = (h - mu_ln) / std_ln * self.ln_gamma + self.ln_beta
        return np.maximum(0, h)                           # ReLU

    def _run_lstm(self, seq, cells):
        """Run a stack of LSTM cells over a sequence. Returns last h."""
        T = seq.shape[0]
        h = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]
        c = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]
        current_seq = seq
        for layer in range(self.num_layers):
            new_seq = []
            for t in range(T):
                h[layer], c[layer] = cells[layer].forward(
                    current_seq[t], h[layer], c[layer]
                )
                new_seq.append(h[layer].copy())
            current_seq = np.stack(new_seq)   # (T, hidden)
        return h[-1]  # last layer's final hidden state

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, X_seq, training=False):
        """
        Args:
            X_seq : (T, input_dim)  — single sequence
            training: apply dropout when True
        Returns:
            logits: (num_classes,)
        """
        proj = self._project(X_seq)                  # (T, hidden)

        # Bidirectional: run fwd and bwd over projected sequence
        h_fwd = self._run_lstm(proj,        self.fwd_cells)
        h_bwd = self._run_lstm(proj[::-1],  self.bwd_cells)

        # Concatenate final hidden states
        h_cat = np.concatenate([h_fwd, h_bwd])      # (hidden*2,)

        # Classifier head
        fc1 = np.maximum(0, self.W_fc1 @ h_cat + self.b_fc1)  # ReLU
        if training and self.dropout > 0:
            mask = (np.random.rand(*fc1.shape) > self.dropout) / (1 - self.dropout)
            fc1 = fc1 * mask
        logits = self.W_fc2 @ fc1 + self.b_fc2      # (num_classes,)
        return logits

    def predict_batch(self, X, training=False):
        """X: (batch, T, input_dim) → logits: (batch, num_classes)"""
        return np.stack([self.forward(X[i], training=training) for i in range(len(X))])

    # ── Loss ──────────────────────────────────────────────────────────────────
    def cross_entropy_loss(self, logits, y_true):
        """
        Numerically-stable cross-entropy.
        logits: (batch, num_classes), y_true: (batch,) int
        """
        probs = _softmax(logits)
        n = len(y_true)
        log_probs = np.log(probs[np.arange(n), y_true] + 1e-12)
        return -log_probs.mean()

    # ── FedProx proximal term ─────────────────────────────────────────────────
    def proximal_loss(self, global_params: list, mu: float = 0.01) -> float:
        """
        FedProx regulariser: (μ/2) * Σ ‖w_local − w_global‖²
        global_params: list of arrays matching self._all_params()
        """
        prox = 0.0
        for local_p, global_p in zip(self._all_params(), global_params):
            prox += np.sum((local_p - global_p) ** 2)
        return (mu / 2.0) * prox

    # ── Numerical gradient + Adam update ────────────────────────────────────
    def train_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        lr: float = 1e-3,
        global_params: list = None,
        mu: float = 0.01,
    ) -> dict:
        """
        One mini-batch training step with numerical gradients + Adam update.

        Args:
            X_batch      : (batch, T, input_dim)
            y_batch      : (batch,) int labels
            lr           : Adam learning rate
            global_params: global model weights for FedProx (None = plain CE)
            mu           : FedProx proximal coefficient
        Returns:
            dict with 'ce_loss', 'prox_loss', 'total_loss'
        """
        eps = 1e-4
        params = self._all_params()

        # ── Forward pass to get current loss ──────────────────────────────
        logits = self.predict_batch(X_batch, training=True)
        ce     = self.cross_entropy_loss(logits, y_batch)
        prox   = self.proximal_loss(global_params, mu) if global_params else 0.0
        total  = ce + prox

        # ── Numerical gradient (central differences) ─────────────────────
        grads = []
        for p in params:
            g = np.zeros_like(p)
            it = np.nditer(p, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = p[idx]

                p[idx] = orig + eps
                lo_plus = self.cross_entropy_loss(
                    self.predict_batch(X_batch, training=False), y_batch
                )
                if global_params:
                    lo_plus += self.proximal_loss(global_params, mu)

                p[idx] = orig - eps
                lo_minus = self.cross_entropy_loss(
                    self.predict_batch(X_batch, training=False), y_batch
                )
                if global_params:
                    lo_minus += self.proximal_loss(global_params, mu)

                p[idx] = orig
                g[idx] = (lo_plus - lo_minus) / (2 * eps)
                it.iternext()
            grads.append(g)

        # ── Adam update ───────────────────────────────────────────────────
        self._adam_t += 1
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        t = self._adam_t
        for i, (p, g) in enumerate(zip(params, grads)):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * g
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * g ** 2
            m_hat = self._adam_m[i] / (1 - beta1 ** t)
            v_hat = self._adam_v[i] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        return {"ce_loss": float(ce), "prox_loss": float(prox), "total_loss": float(total)}


# ── Utility functions ─────────────────────────────────────────────────────────
def _sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# ── Instantiate model ─────────────────────────────────────────────────────────
ids_model = LSTMIdsModel(
    input_dim=NUM_FEATURES,
    hidden_dim=64,           # reduced for numerical-grad feasibility
    num_classes=NUM_CLASSES,
    num_layers=2,
    dropout=0.3,
    seed=42,
)

total_params = sum(p.size for p in ids_model._all_params())
print("=" * 58)
print("  LSTM-IDS Model (NumPy) — Architecture Summary")
print("=" * 58)
print(f"  Input dim      : {NUM_FEATURES}")
print(f"  Hidden dim     : 64  (bidirectional → 128 at head)")
print(f"  LSTM layers    : 2   (forward + backward each)")
print(f"  Num classes    : {NUM_CLASSES}  ({', '.join(CLASS_NAMES)})")
print(f"  Dropout        : 0.30")
print(f"  Total params   : {total_params:,}")
print("=" * 58)
