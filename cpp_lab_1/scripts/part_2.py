from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
figures_dir = base_dir / "figures"
figures_dir.mkdir(exist_ok=True)

df100 = pd.read_csv(data_dir / "dice100.csv")
print( df100.head())
plt.figure(figsize=(10, 5))
plt.bar(df100["value"], df100["normal"], width=1.0, alpha=0.6, label="Normal")
plt.bar(df100["value"], df100["penalty"], width=1.0, alpha=0.5, label="Penalty")
plt.bar(df100["value"], df100["bonus"], width=1.0, alpha=0.5, label="Bonus")
plt.xlabel("Значение", fontsize=16)
plt.ylabel("Вероятность", fontsize=16)
plt.title("Распределение вероятностей для Dice [1, 100]", fontsize=20)
plt.legend(fontsize=14)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=13)
plt.savefig(figures_dir / "dice100.pdf")
plt.close()

df_pool = pd.read_csv(data_dir / "pool3d6.csv")
plt.figure(figsize=(10, 5))
x = df_pool["value"]
w = 0.28
plt.bar(x - w, df_pool["normal"], width=w, label="Normal")
plt.bar(x, df_pool["penalty"], width=w, label="Penalty")
plt.bar(x + w, df_pool["bonus"], width=w, label="Bonus")
plt.xticks(x)
plt.xlabel("Значение", fontsize=16)
plt.ylabel("Вероятность", fontsize=16)
plt.title("Распределение вероятностей для ThreeDicePool 3x[1, 6]", fontsize=20)
plt.legend(fontsize=14)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=13)
plt.savefig(figures_dir / "pool3d6.pdf")
plt.close()