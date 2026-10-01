import cv2
import trackpy as tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import warnings
import os
import glob

# Отключаем лишние предупреждения
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    # ==========================================
    # НАСТРОЙКИ ПАКЕТНОЙ ОБРАБОТКИ
    # ==========================================
    # Укажи путь к папке, где лежат твои видеофайлы (например, .avi или .mp4)
    VIDEO_FOLDER = r"C:\Users\Admin\Desktop\Milliken"
    
    # Находим все видео файлы в папке
    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.avi"))
    
    if not video_files:
        print(f"В папке {VIDEO_FOLDER} не найдено видеофайлов (.avi / .mp4). Проверь путь!")
        return

    print(f"Найдено видеофайлов для обработки: {len(video_files)}")
    for f in video_files:
        print(f" - {os.path.basename(f)}")

    FPS = 50.0
    pixels_per_mm = None # Масштаб откалибруем по первому видео и применим ко всем

    # Общий список для сбора данных со всех видео для финального графика
    all_physics_data = []

    # ==========================================
    # ЦИКЛ ПО ВСЕМ ВИДЕОФАЙЛАМ
    # ==========================================
    for video_idx, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"\n" + "="*50)
        print(f"ОБРАБОТКА ФАЙЛА [{video_idx+1}/{len(video_files)}]: {video_name}")
        print("="*50)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Ошибка: Не удалось прочитать файл {video_name}. Пропускаем.")
            continue

        # --- ЭТАП 1: КАЛИБРОВКА МАСШТАБА (только для первого видео или если еще не задан) ---
        if pixels_per_mm is None:
            clicks = []
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicks.append((x, y))
                    print(f"Клик {len(clicks)}: (x={x}, y={y})")

            cv2.namedWindow(f"Calibration - {video_name}")
            cv2.setMouseCallback(f"Calibration - {video_name}", mouse_callback)

            print("--- КАЛИБРОВКА МАСШТАБА ---")
            print("1. Кликни на верхнюю полоску.")
            print("2. Кликни на полоску через 4 деления (вниз).")
            print("(После двух кликов нажми 'q')")

            disp = frame.copy()
            while True:
                temp_disp = disp.copy()
                if len(clicks) == 1:
                    cv2.circle(temp_disp, clicks[0], 5, (0, 255, 255), -1)
                elif len(clicks) >= 2:
                    cv2.circle(temp_disp, clicks[0], 5, (0, 255, 255), -1)
                    cv2.circle(temp_disp, clicks[1], 5, (0, 255, 255), -1)
                    cv2.line(temp_disp, clicks[0], clicks[1], (0, 255, 255), 2)
                    
                cv2.imshow(f"Calibration - {video_name}", temp_disp)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

            if len(clicks) < 2:
                print("Калибровка пропущена, используем стандарт 150 px/mm")
                pixels_per_mm = 150.0
            else:
                pixels_per_mm = abs(clicks[1][1] - clicks[0][1])
                print(f"Масштаб зафиксирован: {pixels_per_mm} px = 1 мм (применится ко всем видео)")

        # --- ЭТАП 2: ПОКАДРОВАЯ ОБРАБОТКА ---
        print("Покадровый анализ видео (поиск капель)...")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        f_list = []
        frame_idx = 0
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
                
            green_channel = current_frame[:, :, 1]
            features = tp.locate(green_channel, diameter=11, minmass=1000, invert=False)
            
            if not features.empty:
                features['frame'] = frame_idx
                f_list.append(features)
            
            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"Обработано {frame_idx} / {total_frames} кадров")
                
        cap.release()
        
        if not f_list:
            print(f"В файле {video_name} капли не найдены. Пропускаем.")
            continue

        f = pd.concat(f_list, ignore_index=True)

        # --- ЭТАП 3: СВЯЗЫВАНИЕ ТРАЕКТОРИЙ ---
        print("Связывание траекторий...")
        t = tp.link(f, search_range=15, memory=3)
        t1 = tp.filter_stubs(t, 50)
        print(f"Найдено стабильных траекторий в {video_name}: {t1['particle'].nunique()}")

        # График 1: Траектории для текущего видео
        cap = cv2.VideoCapture(video_path)
        _, bg_frame = cap.read()
        cap.release()

        fig, ax = plt.subplots(figsize=(10, 5))
        tp.plot_traj(t1, superimpose=bg_frame[:, :, 1], ax=ax, plot_style={'linewidth': 1, 'alpha': 0.7})
        ax.set_title(f"Траектории: {video_name}")
        plt.show()

        # --- ЭТАП 4: АНАЛИЗ СКОРОСТЕЙ ---
        print("Расчет скоростей падения и подъема...")
        fall_data = []
        rise_data = []

        for pid, trj in t1.groupby('particle'):
            t_sec = trj['frame'].values / FPS
            y_mm = trj['y'].values / pixels_per_mm 
            
            if len(y_mm) < 15:
                continue
                
            y_smooth = scipy.signal.savgol_filter(y_mm, window_length=15, polyorder=2)
            v_inst = np.gradient(y_smooth, t_sec)
            v_threshold = 0.02 
            
            v_fall_vals = v_inst[v_inst > v_threshold]       
            v_rise_vals = np.abs(v_inst[v_inst < -v_threshold]) 
            
            if len(v_fall_vals) > 10:
                fall_data.append({'ID': pid, 'v_mm_s': np.median(v_fall_vals)})
            if len(v_rise_vals) > 10:
                rise_data.append({'ID': pid, 'v_mm_s': np.median(v_rise_vals)})

        df_fall = pd.DataFrame(fall_data)
        df_rise = pd.DataFrame(rise_data)

        # График 2: Скорости для текущего видео
        fig, ax = plt.subplots(figsize=(10, 4))
        if not df_fall.empty:
            ax.scatter(df_fall['ID'], df_fall['v_mm_s'], color='blue', label='Падение', alpha=0.7)
        if not df_rise.empty:
            ax.scatter(df_rise['ID'], df_rise['v_mm_s'], color='red', label='Подъем', alpha=0.7)
        ax.set_title(f"Скорости: {video_name}")
        ax.set_xlabel("Particle ID")
        ax.set_ylabel("Скорость (мм/с)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.show()

        # --- ЭТАП 5: РАСЧЕТ ЗАРЯДА ДЛЯ ТЕКУЩЕГО ВИДЕО ---
        df_physics = pd.merge(df_fall, df_rise, on='ID', suffixes=('_fall', '_rise'))
        
        if len(df_physics) > 0:
            v_fall_m = df_physics['v_mm_s_fall'].values * 1e-3
            v_rise_m = df_physics['v_mm_s_rise'].values * 1e-3
            
            U = 230.0               # Напряжение, В
            l = 0.725e-3            # Расстояние между пластинами, м 
            eta = 1.85e-5           # Вязкость воздуха, Па*с
            rho = 898.0             # Эффективная плотность, кг/м^3
            g = 9.81                # Ускорение свободного падения, м/с^2
            e_charge = 1.602e-19    # Справочный заряд, Кл
            
            r = np.sqrt((9 * eta * v_fall_m) / (2 * g * rho))
            drag_term = 6 * np.pi * eta * r * v_rise_m
            mg_term = (4.0 / 3.0) * np.pi * (r**3) * rho * g
            q = (l / U) * (drag_term + mg_term)
            
            n = q / e_charge
            df_physics['q'] = q
            df_physics['n'] = n
            df_physics['n_rounded'] = np.round(n)
            
            # Фильтрация шумов
            df_physics['error'] = np.abs(df_physics['n'] - df_physics['n_rounded'])
            df_physics = df_physics[df_physics['error'] < 0.15] 
            df_physics = df_physics[(df_physics['n'] >= 0.5) & (df_physics['n'] <= 15)]
            
            if not df_physics.empty:
                # Добавляем в общий список всех видео
                all_physics_data.append(df_physics)
                print(f"Успешно добавлено капель в копилку: {len(df_physics)}")

    # ==========================================
    # ФИНАЛ: ОБЪЕДИНЕННЫЙ ГРАФИК СО ВСЕХ ВИДЕО
    # ==========================================
    if all_physics_data:
        print("\n" + "="*50)
        print("ПОСТРОЕНИЕ СВОДНОГО ИТОГОВОГО ГРАФИКА СО ВСЕХ ВИДЕО...")
        print("="*50)
        
        # Объединяем данные со всех видео в один большой датафрейм
        df_total = pd.concat(all_physics_data, ignore_index=True)
        
        # Сортируем по общему заряду для формирования красивых ступенек
        df_total = df_total.sort_values('q').reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        q_scaled = df_total['q'] * 1e19
        
        scatter = ax.scatter(df_total.index, q_scaled, 
                             c=df_total['n_rounded'], cmap='tab10', s=45, alpha=0.9)
        
        # Линии идеальных уровней
        max_n = int(df_total['n_rounded'].max()) if not df_total.empty else 5
        for i in range(1, max_n + 1):
            level = i * e_charge * 1e19
            ax.axhline(level, color='gray', linestyle='--', alpha=0.4)
            ax.text(-3, level + 0.1, f"{level:.2f}", color='gray', fontsize=9, fontweight='bold')
                             
        ax.set_xlabel("Общий индекс капель (все видео)", fontsize=12)
        ax.set_ylabel("Заряд в Кл * 10^-19", fontsize=12)
        ax.set_title("СВОДНЫЙ РЕЗУЛЬТАТ: Квантование заряда по всем видео", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Заряд капли (n электронов)')
        
        plt.show()
        
        # Итоговый расчет
        e_exp_final = np.median(df_total['q'] / df_total['n_rounded'])
        print(f"\n================ ОБЩИЕ ИТОГИ ЭКСПЕРИМЕНТА ================")
        print(f"Всего обработано видеофайлов:       {len(video_files)}")
        print(f"Всего капель вошло в итоговый анализ: {len(df_total)}")
        print(f"Справочный заряд электрона (e):     {e_charge:.3e} Кл")
        print(f"Экспериментальный заряд (e):        {e_exp_final:.3e} Кл")
        err = abs(e_exp_final - e_charge) / e_charge * 100
        print(f"Итоговая погрешность:               {err:.1f}%")
        print("==========================================================")
    else:
        print("\nНе удалось собрать данные ни с одного видео для финального графика.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()