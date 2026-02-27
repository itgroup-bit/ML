
import numpy as np
import time

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS        = 15          # local training epochs
BATCH_SIZE    = 32          # mini-batch size
LEARNING_RATE = 5e-3        # Adam LR
FEDPROX_MU    = 0.01        # FedProx proximal coefficient
VAL_SPLIT     = 0.15        # fraction held out for validation
SEED          = 42

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight LSTM-IDS with Analytical BPTT Gradients
#
#  Architecture:
#    (batch, seq_len, 78)
#      → Input projection : Linear(78 → H) + LayerNorm + ReLU
#      → UniLSTM (1 layer for BPTT tractability, bidirectional via 2 passes)
#      → Classifier head  : Linear(2H → H) + ReLU + Linear(H → 5)
#
#  Training:
#    Loss:      Cross-Entropy
#    Optimizer: Adam (analytical gradients via BPTT)
#    FedProx:   proximal_loss = (μ/2) * Σ ‖w_local − w_global‖²
# ─────────────────────────────────────────────────────────────────────────────

H = 32        # hidden dim  (bidirectional → 64 at head)
D = NUM_FEATURES
C = NUM_CLASSES

# ── Activation helpers ────────────────────────────────────────────────────────
def _sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def _relu(x):
    return np.maximum(0.0, x)

def _relu_grad(x):
    return (x > 0).astype(float)

# ── Parameter initialisation ──────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def _xavier(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_out, fan_in))

# Input projection  (D → H)
W_proj  = _xavier(D, H);  b_proj  = np.zeros(H)
ln_g    = np.ones(H);      ln_b    = np.zeros(H)

# Forward LSTM (W matrices combined: [f,i,g,o], input = H after proj)
Wh_f = _xavier(H, 4*H);  Wx_f = _xavier(H, 4*H);  b_f = np.zeros(4*H)

# Backward LSTM
Wh_b = _xavier(H, 4*H);  Wx_b = _xavier(H, 4*H);  b_b = np.zeros(4*H)

# Classifier head  (2H → H → C)
W1 = _xavier(2*H, H);  b1 = np.zeros(H)
W2 = _xavier(H,  C);   b2 = np.zeros(C)

def all_params():
    return [W_proj, b_proj, ln_g, ln_b,
            Wh_f, Wx_f, b_f,
            Wh_b, Wx_b, b_b,
            W1, b1, W2, b2]

total_params = sum(p.size for p in all_params())

# Adam moments
ms = [np.zeros_like(p) for p in all_params()]
vs = [np.zeros_like(p) for p in all_params()]
adam_t = 0

# ── Forward: single sample, returns (logits, cache) ──────────────────────────
def forward_one(x_seq):
    """x_seq: (T, D)"""
    T = x_seq.shape[0]

    # Input projection + LayerNorm + ReLU
    z_proj = x_seq @ W_proj.T + b_proj          # (T, H)
    mu_ln  = z_proj.mean(-1, keepdims=True)
    std_ln = z_proj.std(-1, keepdims=True) + 1e-8
    z_ln   = (z_proj - mu_ln) / std_ln * ln_g + ln_b
    z_act  = _relu(z_ln)                         # (T, H)

    # Forward LSTM
    h_f = np.zeros((T+1, H)); c_f = np.zeros((T+1, H))
    gates_f = np.zeros((T, 4*H)); cs_f = np.zeros((T, H)); hs_f = np.zeros((T, H))
    for t in range(T):
        g = z_act[t] @ Wx_f.T + h_f[t] @ Wh_f.T + b_f   # (4H,)
        fs = _sigmoid(g[:H]); ii = _sigmoid(g[H:2*H])
        gg = np.tanh(g[2*H:3*H]); oo = _sigmoid(g[3*H:])
        c_f[t+1] = fs*c_f[t] + ii*gg
        h_f[t+1] = oo*np.tanh(c_f[t+1])
        gates_f[t] = g; cs_f[t] = c_f[t+1]; hs_f[t] = h_f[t+1]

    # Backward LSTM (reversed)
    h_bk = np.zeros((T+1, H)); c_bk = np.zeros((T+1, H))
    gates_b = np.zeros((T, 4*H)); cs_b = np.zeros((T, H)); hs_b = np.zeros((T, H))
    for t in range(T-1, -1, -1):
        t_rev = T-1-t
        g = z_act[t] @ Wx_b.T + h_bk[t_rev] @ Wh_b.T + b_b
        fs = _sigmoid(g[:H]); ii = _sigmoid(g[H:2*H])
        gg = np.tanh(g[2*H:3*H]); oo = _sigmoid(g[3*H:])
        c_bk[t_rev+1] = fs*c_bk[t_rev] + ii*gg
        h_bk[t_rev+1] = oo*np.tanh(c_bk[t_rev+1])
        gates_b[t_rev] = g; cs_b[t_rev] = c_bk[t_rev+1]; hs_b[t_rev] = h_bk[t_rev+1]

    # Concatenate last hidden states
    h_cat = np.concatenate([h_f[T], h_bk[T]])   # (2H,)

    # Classifier head
    fc1_pre  = W1 @ h_cat + b1                  # (H,)
    fc1_act  = _relu(fc1_pre)
    logits   = W2 @ fc1_act + b2                # (C,)

    cache = (x_seq, z_proj, mu_ln, std_ln, z_act,
             h_f, c_f, gates_f, cs_f, hs_f,
             h_bk, c_bk, gates_b, cs_b, hs_b,
             h_cat, fc1_pre, fc1_act, logits)
    return logits, cache


# ── Batch forward ─────────────────────────────────────────────────────────────
def forward_batch(X, training=False):
    """X: (B, T, D) → logits: (B, C)"""
    B = X.shape[0]
    logits_all = np.zeros((B, C))
    caches = []
    for i in range(B):
        lgt, cache = forward_one(X[i])
        logits_all[i] = lgt
        caches.append(cache)
    return logits_all, caches


# ── Cross-entropy loss ────────────────────────────────────────────────────────
def ce_loss(logits, y):
    probs = _softmax(logits)
    n = len(y)
    return -np.log(probs[np.arange(n), y] + 1e-12).mean(), probs


# ── FedProx proximal term ─────────────────────────────────────────────────────
def proximal_loss(global_params, mu=0.01):
    """(μ/2) * Σ ‖w_local − w_global‖²"""
    prox = 0.0
    for lp, gp in zip(all_params(), global_params):
        prox += np.sum((lp - gp)**2)
    return (mu / 2.0) * prox


# ── Analytical BPTT gradient for one sample ──────────────────────────────────
def backward_one(cache, d_logits):
    """Compute parameter gradients for one sample via BPTT."""
    (x_seq, z_proj, mu_ln, std_ln, z_act,
     h_f, c_f, gates_f, cs_f, hs_f,
     h_bk, c_bk, gates_b, cs_b, hs_b,
     h_cat, fc1_pre, fc1_act, logits) = cache

    T = x_seq.shape[0]

    # ── Classifier head grads ─────────────────────────────────────────────
    dW2   = np.outer(d_logits, fc1_act)
    db2   = d_logits.copy()
    d_fc1 = W2.T @ d_logits * _relu_grad(fc1_pre)
    dW1   = np.outer(d_fc1, h_cat)
    db1   = d_fc1.copy()
    d_hcat = W1.T @ d_fc1

    d_hf_last  = d_hcat[:H]
    d_hbk_last = d_hcat[H:]

    # ── BPTT through forward LSTM ─────────────────────────────────────────
    dWx_f = np.zeros_like(Wx_f); dWh_f = np.zeros_like(Wh_f); db_f_g = np.zeros(4*H)
    d_z_act_f = np.zeros((T, H))
    dh_next = d_hf_last; dc_next = np.zeros(H)

    for t in range(T-1, -1, -1):
        g = gates_f[t]
        f = _sigmoid(g[:H]); ii = _sigmoid(g[H:2*H])
        gg = np.tanh(g[2*H:3*H]); oo = _sigmoid(g[3*H:])
        c_curr = cs_f[t]
        h_curr = hs_f[t]

        do   = dh_next * np.tanh(c_curr)
        dtc  = dh_next * oo * (1 - np.tanh(c_curr)**2) + dc_next
        df   = dtc * c_f[t]
        di   = dtc * gg
        dg   = dtc * ii

        d_gates = np.concatenate([
            df  * f  * (1 - f),
            di  * ii * (1 - ii),
            dg  * (1 - gg**2),
            do  * oo * (1 - oo),
        ])
        dWx_f += np.outer(d_gates, z_act[t] if z_act is not None else np.zeros(H))
        dWh_f += np.outer(d_gates, h_f[t])
        db_f_g += d_gates
        d_z_act_f[t] += Wx_f.T @ d_gates
        dc_next = dtc * f
        dh_next = Wh_f.T @ d_gates

    # ── BPTT through backward LSTM ────────────────────────────────────────
    dWx_b = np.zeros_like(Wx_b); dWh_b = np.zeros_like(Wh_b); db_b_g = np.zeros(4*H)
    d_z_act_b = np.zeros((T, H))
    dh_next = d_hbk_last; dc_next = np.zeros(H)

    for t_rev in range(T-1, -1, -1):
        t = T-1-t_rev
        g = gates_b[t_rev]
        f = _sigmoid(g[:H]); ii = _sigmoid(g[H:2*H])
        gg = np.tanh(g[2*H:3*H]); oo = _sigmoid(g[3*H:])
        c_curr = cs_b[t_rev]

        do   = dh_next * np.tanh(c_curr)
        dtc  = dh_next * oo * (1 - np.tanh(c_curr)**2) + dc_next
        df   = dtc * c_bk[t_rev]
        di   = dtc * gg
        dg_g = dtc * ii

        d_gates = np.concatenate([
            df  * f  * (1 - f),
            di  * ii * (1 - ii),
            dg_g * (1 - gg**2),
            do  * oo * (1 - oo),
        ])
        dWx_b += np.outer(d_gates, z_act[t])
        dWh_b += np.outer(d_gates, h_bk[t_rev])
        db_b_g += d_gates
        d_z_act_b[t] += Wx_b.T @ d_gates
        dc_next = dtc * f
        dh_next = Wh_b.T @ d_gates

    # ── Grad through ReLU → LayerNorm → Input Projection ─────────────────
    d_z_act = d_z_act_f + d_z_act_b            # (T, H) combined
    d_z_ln  = d_z_act * _relu_grad(z_act)       # through ReLU

    # LayerNorm backward
    z_c = z_proj - mu_ln                        # (T, H)
    inv_std = 1.0 / std_ln
    d_ln_g  = (d_z_ln * z_c * inv_std).sum(0)  # (H,)
    d_ln_b  = d_z_ln.sum(0)                    # (H,)
    d_zhat  = d_z_ln * ln_g
    T_ax    = z_proj.shape[0]
    d_z_proj = inv_std * (
        d_zhat - d_zhat.mean(-1, keepdims=True)
        - z_c * inv_std * (d_zhat * z_c * inv_std).mean(-1, keepdims=True)
    )

    # Input projection backward
    dW_proj = d_z_proj.T @ x_seq               # (H, D)
    db_proj = d_z_proj.sum(0)

    return [dW_proj, db_proj, d_ln_g, d_ln_b,
            dWh_f, dWx_f, db_f_g,
            dWh_b, dWx_b, db_b_g,
            dW1, db1, dW2, db2]


# ── Adam update ───────────────────────────────────────────────────────────────
def adam_update(grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    global adam_t
    adam_t += 1
    t = adam_t
    for i, (p, g) in enumerate(zip(all_params(), grads)):
        ms[i] = beta1 * ms[i] + (1-beta1) * g
        vs[i] = beta2 * vs[i] + (1-beta2) * g**2
        mh = ms[i] / (1 - beta1**t)
        vh = vs[i] / (1 - beta2**t)
        p -= lr * mh / (np.sqrt(vh) + eps)


# ── One training step ─────────────────────────────────────────────────────────
def train_step(X_batch, y_batch, global_params_snap, mu, lr):
    B = len(y_batch)

    # Forward
    logits_all, caches = forward_batch(X_batch, training=True)
    loss_ce, probs = ce_loss(logits_all, y_batch)
    loss_prox = proximal_loss(global_params_snap, mu)
    loss_total = loss_ce + loss_prox

    # dL/d_logits per sample (softmax cross-entropy gradient)
    d_logits = probs.copy()
    d_logits[np.arange(B), y_batch] -= 1.0
    d_logits /= B

    # Accumulate BPTT gradients
    grads = [np.zeros_like(p) for p in all_params()]
    for i in range(B):
        g_i = backward_one(caches[i], d_logits[i])
        for j in range(len(grads)):
            grads[j] += g_i[j]

    # FedProx gradient contribution: dProx/dw = mu * (w - w_global)
    for j, (lp, gp) in enumerate(zip(all_params(), global_params_snap)):
        grads[j] += mu * (lp - gp)

    # Clip gradients for stability
    for g in grads:
        np.clip(g, -5.0, 5.0, out=g)

    adam_update(grads, lr)

    return float(loss_ce), float(loss_prox), float(loss_total)


# ── Accuracy helpers ──────────────────────────────────────────────────────────
def predict_acc(X, y):
    logits, _ = forward_batch(X)
    preds = logits.argmax(1)
    return float((preds == y).mean())

def per_class_acc(X, y):
    logits, _ = forward_batch(X)
    preds = logits.argmax(1)
    results = {}
    for c in range(NUM_CLASSES):
        mask = y == c
        if mask.sum() == 0:
            results[CLASS_NAMES[c]] = "N/A"
        else:
            results[CLASS_NAMES[c]] = f"{(preds[mask]==c).mean():.2%}  ({mask.sum()} samples)"
    return results


# ── Train / val split ─────────────────────────────────────────────────────────
n_total = client_X.shape[0]
n_val   = max(1, int(n_total * VAL_SPLIT))
n_train = n_total - n_val
perm    = np.random.permutation(n_total)
X_train, y_train = client_X[perm[:n_train]], client_y[perm[:n_train]]
X_val,   y_val   = client_X[perm[n_train:]], client_y[perm[n_train:]]
print(f"Train: {n_train}  |  Val: {n_val}")
print(f"Total params: {total_params:,}")

# Snapshot global weights for FedProx
global_params_snapshot = [p.copy() for p in all_params()]

# Baseline accuracy
baseline_val_acc = predict_acc(X_val, y_val)
print(f"\nBaseline val acc (random init): {baseline_val_acc:.2%}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
history = {"epoch": [], "ce_loss": [], "prox_loss": [], "total_loss": [],
           "train_acc": [], "val_acc": []}

print(f"{'Epoch':>6}  {'CE Loss':>9}  {'Prox Loss':>10}  {'Total Loss':>11}  {'Train Acc':>10}  {'Val Acc':>8}")
print("-" * 65)

t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    idx = np.random.permutation(n_train)
    Xs, Ys = X_train[idx], y_train[idx]

    ep_ce = []; ep_prox = []; ep_total = []
    for start in range(0, n_train, BATCH_SIZE):
        xb = Xs[start:start+BATCH_SIZE]
        yb = Ys[start:start+BATCH_SIZE]
        ce, prox, total = train_step(xb, yb, global_params_snapshot, FEDPROX_MU, LEARNING_RATE)
        ep_ce.append(ce); ep_prox.append(prox); ep_total.append(total)

    mce    = float(np.mean(ep_ce))
    mprox  = float(np.mean(ep_prox))
    mtotal = float(np.mean(ep_total))
    tacc   = predict_acc(X_train, y_train)
    vacc   = predict_acc(X_val, y_val)

    history["epoch"].append(epoch)
    history["ce_loss"].append(mce)
    history["prox_loss"].append(mprox)
    history["total_loss"].append(mtotal)
    history["train_acc"].append(tacc)
    history["val_acc"].append(vacc)

    print(f"{epoch:>6}  {mce:>9.4f}  {mprox:>10.6f}  {mtotal:>11.4f}  {tacc:>10.2%}  {vacc:>8.2%}")

t_elapsed = time.time() - t_start

# ── Final metrics summary ─────────────────────────────────────────────────────
final_val_acc   = history["val_acc"][-1]
final_train_acc = history["train_acc"][-1]
final_ce_loss   = history["ce_loss"][-1]
final_prox      = history["prox_loss"][-1]
final_total     = history["total_loss"][-1]
best_val_acc    = max(history["val_acc"])
best_epoch      = history["epoch"][history["val_acc"].index(best_val_acc)]

print()
print("=" * 65)
print("  Local Training Complete — Final Metrics")
print("=" * 65)
print(f"  Epochs trained   : {EPOCHS}")
print(f"  Training time    : {t_elapsed:.1f}s")
print(f"  Final CE loss    : {final_ce_loss:.4f}")
print(f"  Final Prox loss  : {final_prox:.6f}  (μ={FEDPROX_MU})")
print(f"  Final Total loss : {final_total:.4f}")
print(f"  Train accuracy   : {final_train_acc:.2%}")
print(f"  Val   accuracy   : {final_val_acc:.2%}")
print(f"  Best val acc     : {best_val_acc:.2%}  (epoch {best_epoch})")
print("-" * 65)
print("  Per-class Validation Accuracy:")
for cls, info in per_class_acc(X_val, y_val).items():
    print(f"    {cls:<15} : {info}")
print("=" * 65)

# ── Metrics dict for downstream ───────────────────────────────────────────────
training_metrics = {
    "ce_loss":      final_ce_loss,
    "prox_loss":    final_prox,
    "total_loss":   final_total,
    "train_acc":    final_train_acc,
    "val_acc":      final_val_acc,
    "best_val_acc": best_val_acc,
    "best_epoch":   best_epoch,
    "epochs":       EPOCHS,
    "fedprox_mu":   FEDPROX_MU,
    "history":      history,
}
