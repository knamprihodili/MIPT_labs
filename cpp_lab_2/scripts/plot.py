from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
figures_dir = base_dir / "figures"
figures_dir.mkdir(exist_ok=True)

df_n = pd.read_csv(data_dir / "bench_n.csv")
plt.figure(figsize=(10, 5))
plt.plot(df_n["n"], df_n["insert_ns"], marker="o", label="Insert")
plt.plot(df_n["n"], df_n["find_ns"], marker="s", label="Find")
plt.plot(df_n["n"], df_n["erase_ns"], marker="^", label="Erase")
plt.xscale("log")
plt.xlabel("Количество элементов N", fontsize=16)
plt.ylabel("Время операции, нс", fontsize=16)
plt.title("Зависимость времени операций от размера списка (p = 0.5)", fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=13)
plt.savefig(figures_dir / "bench_n.pdf")
plt.close()

df_p = pd.read_csv(data_dir / "bench_p.csv")
plt.figure(figsize=(10, 5))
plt.plot(df_p["p"], df_p["insert_ns"], marker="o", label="Insert")
plt.plot(df_p["p"], df_p["find_ns"], marker="s", label="Find")
plt.plot(df_p["p"], df_p["erase_ns"], marker="^", label="Erase")
plt.xlabel("Вероятность p", fontsize=16)
plt.ylabel("Время операции, нс", fontsize=16)
plt.title("Зависимость времени операций от вероятности p (N = 100000)", fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=13)
plt.savefig(figures_dir / "bench_p.pdf")
plt.close()