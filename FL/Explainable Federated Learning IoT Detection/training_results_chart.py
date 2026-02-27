
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Zerve design system ───────────────────────────────────────────────────────
BG       = "#1D1D20"
TXT      = "#fbfbff"
SECONDARY = "#909094"
BLUE     = "#A1C9F4"
ORANGE   = "#FFB482"
GREEN    = "#8DE5A1"
CORAL    = "#FF9F9B"
LAVENDER = "#D0BBFF"
YELLOW   = "#ffd400"

epochs = history["epoch"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# ── Plot 1: Loss curves ───────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)

ax.plot(epochs, history["ce_loss"],    color=BLUE,    lw=2.0, label="CE Loss",    marker="o", markersize=4)
ax.plot(epochs, history["prox_loss"],  color=ORANGE,  lw=2.0, label="Prox Loss (FedProx μ=0.01)", marker="s", markersize=4)
ax.plot(epochs, history["total_loss"], color=YELLOW,  lw=2.5, label="Total Loss", marker="^", markersize=4, linestyle="--")

ax.set_title("Training Loss Curves", color=TXT, fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Epoch", color=SECONDARY, fontsize=11)
ax.set_ylabel("Loss", color=SECONDARY, fontsize=11)
ax.tick_params(colors=SECONDARY)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(frameon=True, facecolor="#2a2a2e", edgecolor="#444", labelcolor=TXT, fontsize=9)
ax.grid(axis="y", color="#333", linewidth=0.5, alpha=0.7)

# ── Plot 2: Accuracy curves ───────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(BG)

ax.plot(epochs, [v*100 for v in history["train_acc"]], color=GREEN,   lw=2.0, label="Train Accuracy", marker="o", markersize=4)
ax.plot(epochs, [v*100 for v in history["val_acc"]],   color=LAVENDER, lw=2.0, label="Val Accuracy",   marker="s", markersize=4)

# Highlight best val epoch
best_ep = best_epoch
best_va = best_val_acc * 100
ax.axvline(best_ep, color=YELLOW, linestyle=":", alpha=0.6, linewidth=1.5)
ax.annotate(f"Best val\n{best_va:.1f}% @ ep {best_ep}",
            xy=(best_ep, best_va), xytext=(best_ep+0.8, best_va-5),
            color=YELLOW, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2))

ax.set_ylim(50, 105)
ax.set_title("Accuracy over Epochs", color=TXT, fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Epoch", color=SECONDARY, fontsize=11)
ax.set_ylabel("Accuracy (%)", color=SECONDARY, fontsize=11)
ax.tick_params(colors=SECONDARY)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(frameon=True, facecolor="#2a2a2e", edgecolor="#444", labelcolor=TXT, fontsize=9)
ax.grid(axis="y", color="#333", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.suptitle("LSTM-IDS   ·   Client 0   ·   FedProx (μ=0.01)",
             color=TXT, fontsize=13, fontweight="bold", y=1.02)

training_chart = fig
plt.show()

# ── Print summary metrics ─────────────────────────────────────────────────────
print(f"  CE loss  {final_ce_loss:.4f}  |  Prox loss  {final_prox:.5f}  |  Total  {final_total:.4f}")
print(f"  Train acc  {final_train_acc:.2%}  |  Val acc  {final_val_acc:.2%}  |  Best val  {best_val_acc:.2%} (ep {best_epoch})")
