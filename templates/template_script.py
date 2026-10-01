import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. НАСТРОЙКИ ПУТЕЙ ---
# Определяем путь к папке figures относительно папки scripts (на уровень выше)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True) # Создаст папку, если ее вдруг нет

# --- 2. МАТЕМАТИЧЕСКИЕ ФУНКЦИИ ---
def linear_regression(x, y):
    """
    Метод наименьших квадратов для y = b*x + a.
    Возвращает: a (сдвиг), b (наклон), a_err, b_err
    """
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

def linearize(x, y, x_err=0, y_err=0, mode='none'):
    """
    Преобразование координат для линеаризации графика.
    Поддерживаемые режимы (mode):
    - 'none': без изменений (y = kx + b)
    - 'log-log': степенная зависимость (ln(y) от ln(x))
    - 'semi-log': экспоненциальная (ln(y) от x)
    - 'inverse_x': обратная пропорциональность (y от 1/x)
    """
    if mode == 'log-log':
        return np.log(x), np.log(y), x_err/x, y_err/y
    elif mode == 'semi-log':
        return x, np.log(y), x_err, y_err/y
    elif mode == 'inverse_x':
        return 1/x, y, x_err/(x**2), y_err
    
    return x, y, x_err, y_err

# --- 3. ВВОД ДАННЫХ (АГЕНТ ЗАПОЛНЯЕТ ЗДЕСЬ) ---
# Пример данных
x_raw = np.array([1.0, 2.0, 3.0, 4.0])
y_raw = np.array([2.1, 4.0, 5.9, 8.2])
x_err_raw = 0.1
y_err_raw = 0.2

# Выбор режима обработки
MODE = 'none' # Агент может поменять на 'log-log', 'inverse_x' и т.д.
x_data, y_data, x_err, y_err = linearize(x_raw, y_raw, x_err_raw, y_err_raw, mode=MODE)

# --- 4. РАСЧЕТЫ ---
a, b, a_err, b_err = linear_regression(x_data, y_data)

print(f"--- РЕЗУЛЬТАТЫ (Режим: {MODE}) ---")
print(f"Наклон (k): {b:.4f} ± {b_err:.4f}")
print(f"Сдвиг  (b): {a:.4f} ± {a_err:.4f}")

# --- 5. ПОСТРОЕНИЕ ГРАФИКА ---
plt.figure(figsize=(10, 6))

# Экспериментальные точки
plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', color='red', 
             capsize=5, capthick=2, elinewidth=2, markersize=8, 
             label='Эксперимент', zorder=3)

# Линия аппроксимации
x_line = np.linspace(min(x_data) - 0.1*abs(min(x_data)), max(x_data) + 0.1*abs(max(x_data)), 100)
y_line = b * x_line + a
plt.plot(x_line, y_line, color='blue', linewidth=2.5, linestyle='--',
         label=f'МНК: $y = {b:.3f}x {a:+.3f}$', zorder=2)

# Оформление (Агент подставит нужные подписи)
plt.xlabel('Ось X', fontsize=18)
plt.ylabel('Ось Y', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Сохранение файла
output_filename = 'graph_1.pdf' # Агент изменит имя под конкретный график
save_path = os.path.join(FIGURES_DIR, output_filename)
plt.savefig(save_path, bbox_inches='tight', facecolor='white')

print(f"График успешно сохранен: {save_path}")