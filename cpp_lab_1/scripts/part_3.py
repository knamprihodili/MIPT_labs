from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
figures_dir = base_dir / "figures"
figures_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / "double_dice.csv")

plt.figure(figsize=(10, 5))
plt.bar(df["value"], df["prob"], width=1.0, alpha=0.75, label="DoubleDice")
plt.xlabel("Значение", fontsize=16)
plt.ylabel("Вероятность", fontsize=16)
plt.title("Распределение вероятностей для DoubleDice [1, 100]", fontsize=20)
plt.legend(fontsize=14)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=13)
plt.savefig(figures_dir / "double_dice.pdf")
plt.close()