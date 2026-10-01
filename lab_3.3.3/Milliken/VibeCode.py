import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
# ==========================================
# НАСТРОЙКИ
# ==========================================
VIDEO_PATH = r"C:\Users\Admin\Desktop\Milliken\test.avi"

clicks = []
current_setup_frame = None
start_frame_idx = 0

def mouse_callback(event, x, y, flags, param):
    """Обработчик кликов мыши"""
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Клик {len(clicks)}: (x={x}, y={y})")

def on_trackbar(val):
    """Срабатывает при перемещении ползунка времени"""
    global current_setup_frame, start_frame_idx, cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        current_setup_frame = frame.copy()
        start_frame_idx = val

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 50.0

# --- ЭТАП 1: ИНТЕРАКТИВНАЯ КАЛИБРОВКА С ВЫБОРОМ ВРЕМЕНИ ---
cv2.namedWindow("Setup")
cv2.setMouseCallback("Setup", mouse_callback)

if total_frames > 0:
    cv2.createTrackbar("Start Frame", "Setup", 0, total_frames - 1, on_trackbar)
on_trackbar(0) 

print("--- КАЛИБРОВКА И ВЫБОР ВРЕМЕНИ ---")
print("0. ПРОКРУТИ ползунок 'Start Frame' до момента появления нужной капли.")
print("1. Кликни на верхнюю полоску.")
print("2. Кликни на полоску через 4 деления (вниз).")
print("3. Кликни в центр масляной капли.")
print("(Если ошибся, нажми 'r' для сброса. Для выхода нажми 'q')")

while True:
    if current_setup_frame is None:
        break
        
    disp = current_setup_frame.copy()
    current_time_sec = start_frame_idx / fps
    
    cv2.putText(disp, f"Time: {current_time_sec:.2f} s", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if len(clicks) == 0:
        cv2.putText(disp, "Step 1: Click 1st stripe", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif len(clicks) == 1:
        cv2.circle(disp, clicks[0], 5, (0, 255, 255), -1)
        cv2.putText(disp, "Step 2: Click 2nd stripe (1mm)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif len(clicks) >= 2:
        cv2.circle(disp, clicks[0], 5, (0, 255, 255), -1)
        cv2.circle(disp, clicks[1], 5, (0, 255, 255), -1)
        cv2.line(disp, clicks[0], clicks[1], (0, 255, 255), 2)
        cv2.putText(disp, "Step 3: Click on the DROP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    if len(clicks) >= 3:
        cv2.circle(disp, clicks[2], 5, (0, 0, 255), -1)
        cv2.imshow("Setup", disp)
        cv2.waitKey(500) 
        break

    cv2.imshow("Setup", disp)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('r'):
        clicks = [] 
        print("Клики сброшены.")
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Setup")

y1 = clicks[0][1]
y2 = clicks[1][1]
PIXELS_PER_4_STRIPES = abs(y2 - y1)
print(f"\nМасштаб установлен: {PIXELS_PER_4_STRIPES} px = 1 мм")

# --- ЭТАП 2: ТРЕКИНГ CSRT В КОРИДОРЕ С АВТОСТОПОМ ---
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
ret, frame = cap.read()

start_x, start_y = clicks[2]
corridor_width = 30  # Ширина коридора влево и вправо (всего 60 px)
boundary_margin = 100 #Отступ от верхнего и нижнего края кадра (в пикселях)

# Границы коридора
roi_x_min = max(0, int(start_x - corridor_width))
roi_x_max = min(frame.shape[1], int(start_x + corridor_width))

# Подготовка начальной рамки для CSRT
box_size = 30 
bbox = (int(start_x - box_size/2), int(start_y - box_size/2), box_size, box_size)
bbox = tuple(map(int, bbox))

tracker = cv2.TrackerCSRT_create()

# Инициализируем трекер на кадре с закрашенными краями
init_frame = frame.copy()
init_frame[:, :roi_x_min] = 0
init_frame[:, roi_x_max:] = 0
tracker.init(init_frame, bbox)

times = []
positions_y = []
frame_idx = start_frame_idx
frame_height = frame.shape[0]

print(f"\nНачинаю трекинг с {start_frame_idx / fps:.2f} сек... (Автостоп у границ экрана включен)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        current_time = frame_idx / fps
        
        # Создаем копию кадра и закрашиваем всё вне коридора черным
        track_frame = frame.copy()
        track_frame[:, :roi_x_min] = 0
        track_frame[:, roi_x_max:] = 0
        
        # Передаем обрезанный кадр в трекер
        success, box = tracker.update(track_frame)
        
        # Отрисовка границ коридора и линий автостопа на оригинальном кадре
        cv2.line(frame, (roi_x_min, 0), (roi_x_min, frame_height), (255, 0, 0), 1)
        cv2.line(frame, (roi_x_max, 0), (roi_x_max, frame_height), (255, 0, 0), 1)
        cv2.line(frame, (0, boundary_margin), (frame.shape[1], boundary_margin), (0, 165, 255), 1)
        cv2.line(frame, (0, frame_height - boundary_margin), (frame.shape[1], frame_height - boundary_margin), (0, 165, 255), 1)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            center_y = y + h / 2.0
            
            # Автоматическая остановка при приближении к краю
            if center_y < boundary_margin or center_y > (frame_height - boundary_margin):
                print(f"Капля достигла границы кадра (y={center_y:.1f}). Трекинг завершен.")
                break
                
            times.append(current_time)
            positions_y.append(center_y)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int(x + w / 2), int(center_y)), 2, (0, 0, 255), -1)
            cv2.putText(frame, f"y: {center_y:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "LOST", (roi_x_min + 5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Corridor Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Трекинг остановлен пользователем.")
            break
            
finally:
    cap.release()
    cv2.destroyAllWindows()

# --- ЭТАП 3: АВТОМАТИЧЕСКИЙ АНАЛИЗ ВСЕХ УЧАСТКОВ ---
import scipy.signal 

print("\nОбработка данных...")
if len(times) > 20:
    PIXELS_PER_MM = PIXELS_PER_4_STRIPES / 1.0 
    METERS_PER_PIXEL = (1e-3) / PIXELS_PER_MM

    t = np.array(times)
    y = np.array(positions_y)

    # 1. Сглаживаем данные, чтобы отсечь микро-дрожания рамки при трекинге
    y_smooth = np.convolve(y, np.ones(11)/11, mode='same')
    
    # 2. Ищем точки разворота (вершины зигзагов на графике)
    # prominence=15 игнорирует случайные шумы менее 15 пикселей
    peaks_bottom, _ = scipy.signal.find_peaks(y_smooth, prominence=3, distance=10)
    peaks_top, _ = scipy.signal.find_peaks(-y_smooth, prominence=3, distance=10)
    
    # Собираем все точки разворота вместе с началом и концом массива
    turn_points = np.sort(np.concatenate(([0], peaks_bottom, peaks_top, [len(y)-1])))
    
    print(f"\n--- Найдено участков движения: {len(turn_points)-1} ---")
    
    plt.figure(figsize=(12, 7))
    # Рисуем все сырые данные тусклым серым цветом для фона
    plt.plot(t, y, color='lightgray', marker='.', linestyle='', alpha=0.5, label='Сырая траектория')

    # Отрезаем по 15 кадров с краев каждого участка, так как в моменты 
    # включения/выключения поля капля движется с ускорением (график кривой), 
    # а нам для МНК нужно только равномерное движение (строго прямая линия).
    margin = 15 

    for i in range(len(turn_points) - 1):
        start_idx = turn_points[i] + margin
        end_idx = turn_points[i+1] - margin
        
        # Обрабатываем только те участки, где после обрезки осталось достаточно точек
        if end_idx - start_idx > 10: 
            t_seg = t[start_idx:end_idx]
            y_seg = y[start_idx:end_idx]
            
            slope, int_val = np.polyfit(t_seg, y_seg, 1)
            v_m = abs(slope) * METERS_PER_PIXEL
            t_1mm = (1e-3) / v_m
            
            # Определяем направление (на графике Y растет при движении вниз по экрану)
            if slope > 0:
                c = 'blue'
                dir_name = "Падение (Без поля)"
            else:
                c = 'red'
                dir_name = "Подъем (В поле)"
                
            print(f"Участок {i+1} | {dir_name}: v = {v_m:.6e} м/с | t на 1мм = {t_1mm:.3f} с")
            
            plt.scatter(t_seg, y_seg, color=c, s=15)
            plt.plot(t_seg, slope * t_seg + int_val, color=c, linewidth=2, 
                     label=f'{dir_name} ({t_1mm:.2f} с)')

    plt.gca().invert_yaxis()
    plt.xlabel("Время t, с")
    plt.ylabel("Координата Y, px")
    plt.title("Автоматическая аппроксимация всех участков траектории")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Группируем легенду, чтобы одинаковые подписи не дублировались
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.show()
else:
    print("Слишком мало данных для анализа.")

