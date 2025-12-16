# plot_training_curves.py
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

central_path = os.path.join(BASE_DIR, "data/central_history.npz")
fed_path     = os.path.join(BASE_DIR, "data/fed_history.npz")

central = np.load(central_path)
fed      = np.load(fed_path)

central_acc = central["val_acc"]
fed_acc     = fed["val_acc"]

epochs = np.arange(1, len(central_acc) + 1)
rounds = np.arange(1, len(fed_acc) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, central_acc, label="Centralized (all data on PC)",
         marker="o", linewidth=1.2)
plt.plot(rounds, fed_acc,     label="Federated (clients + FedAvg)",
         marker="s", linewidth=1.2)

plt.xlabel("Epoch / Round")
plt.ylabel("Validation accuracy")
plt.ylim(0.0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.title("Centralized vs Federated training")
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "../Slides/figures/fig_central_vs_fed.png")
plt.savefig(out_path, dpi=200)
plt.show()

print("已保存对比图:", out_path)
