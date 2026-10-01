import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. НАСТРОЙКИ ПУТЕЙ ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- 2. МАТЕМАТИЧЕСКИЕ ФУНКЦИИ ---
def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    denominator = np.sum((x - x_mean)**2)
    b = np.sum((x - x_mean) * (y - y_mean)) / denominator
    a = y_mean - b * x_mean
    
    rss = np.sum((y - (a + b * x))**2)
    s_squared = rss / (n - 2) if n > 2 else 0
    s = np.sqrt(s_squared)
    
    b_error = s / np.sqrt(denominator) if denominator != 0 else 0
    a_error = s * np.sqrt(1/n + (x_mean**2)/denominator) if denominator != 0 else 0
    
    return a, b, a_error, b_error

# --- 3. ВВОД ДАННЫХ (Таблица А.3) ---
x_data = np.array([0.2, 0.25, 0.3125, 0.4, 0.5, 0.625, 1.0, 1.25, 2.0, 2.5]) # 1/T, кГц
y_data = np.array([0.2, 0.25, 0.31,   0.4, 0.5, 0.625, 1.0, 1.25, 2.0, 2.5]) # delta nu, кГц

# --- 4. РАСЧЕТЫ ---
a, b, a_err, b_err = linear_regression(x_data, y_data)

print(f"--- РЕЗУЛЬТАТЫ (delta nu от 1/T) ---")
print(f"Наклон (k): {b:.4f} ± {b_err:.4f}")
print(f"Сдвиг  (b): {a:.4f} ± {a_err:.4f}")

# --- 5. ПОСТРОЕНИЕ ГРАФИКА ---
plt.figure(figsize=(9, 6), dpi=300)

# Экспериментальные точки
plt.scatter(x_data, y_data, color='#0022ff', s=45, label='Экспериментальные данные', zorder=3)

# Сплошная линия аппроксимации от нуля
x_max = max(x_data) * 1.05
x_line = np.linspace(0, x_max, 200)
y_line = b * x_line + a
plt.plot(x_line, y_line, color='#111111', linewidth=1.8, linestyle='-',
         label=f'МНК: $\\delta\\nu = ({b:.4f} \\pm {b_err:.4f}) \\cdot (1/T) {a:+.4f}$', zorder=2)

# Оформление осей и ограничений
plt.xlim(0, x_max)
plt.ylim(0, max(y_data) * 1.08)

# Увеличенная оцифровка осей
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.xlabel(r'$1/T$, кГц', fontsize=30)
plt.ylabel(r'$\delta\nu$, кГц', fontsize=30)
plt.legend(fontsize=18, frameon=True, edgecolor='#cccccc')
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()

output_filename = 'graph_dist_T.pdf'
save_path = os.path.join(FIGURES_DIR, output_filename)
plt.savefig(save_path, bbox_inches='tight', facecolor='white')

print(f"График успешно сохранен: {save_path}")