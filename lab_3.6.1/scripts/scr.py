import pandas as pd
import matplotlib.pyplot as plt
import os

# Определяем абсолютный путь к папке, где лежит этот скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'r1.csv')
PDF_PATH = os.path.join(SCRIPT_DIR, 'achh_plot.pdf')

# Читаем данные по абсолютному пути
df = pd.read_csv(CSV_PATH, skiprows=10)

plt.figure(figsize=(10, 6))

# Строим график
plt.plot(df['index'], df['CH1_Voltage(mV)'], color='blue', linewidth=0.5)

plt.title('Осциллограмма (огибающая соответствует АЧХ)')
plt.xlabel('Индекс отсчета')
plt.ylabel('Напряжение, мВ')
plt.grid(True)

# Сохраняем рядом со скриптом
plt.savefig(PDF_PATH, format='pdf', bbox_inches='tight')

# Очищаем память
plt.close()