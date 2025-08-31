import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from math import degrees
from PIL import Image, ImageDraw, ImageFont

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# Параметры визуализации
arc_thickness = 3  # Толщина дуги
arc_color = (255, 0, 255)  # Цвет дуги (магента)
arc_radius = 30  # Радиус дуг в пикселях
font_size = 15  # Размер шрифта для углов
text_color = (255, 255, 255)  # Белый цвет для текста углов
status_color = (0, 255, 255)  # Жёлтый цвет для текста кадра и статуса
warning_color = (0, 0, 255)  # Красный цвет для предупреждений

# Функция для вычисления углов поворота векторов для дуги (2D)
def calculate_vector_angle(point1, point2):
    vector = np.array(point2) - np.array(point1)
    angle = degrees(np.arctan2(vector[1], vector[0]))
    return angle

# Функция для отрисовки дуги вокруг сустава
def draw_arc(image, point_a, point_b, point_c, angle_curr, radius, width, height, color):
    center = (int(point_b[0] * width), int(point_b[1] * height))
    angle_ab = calculate_vector_angle(point_b, point_a)
    angle_bc = calculate_vector_angle(point_b, point_c)
    angle_ab = (angle_ab + 360) % 360
    angle_bc = (angle_bc + 360) % 360
    start_angle = min(angle_ab, angle_bc)
    end_angle = max(angle_ab, angle_bc)
    if abs(end_angle - start_angle) > 180:
        start_angle, end_angle = end_angle, start_angle
    if abs(angle_curr) > 0.1:  # Пропускаем нулевые углы
        cv2.ellipse(image, center, (int(radius / max(width, height) * width), int(radius / max(width, height) * height)),
                    0, start_angle, end_angle, color, arc_thickness)
    return start_angle, end_angle

# Функция для наложения текста с поддержкой кириллицы
def put_text_pil(image, text, position, font_size=15, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except IOError:
        print("Не удалось загрузить шрифт Arial, используется шрифт по умолчанию")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image

# Функция для вычисления размеров тела для масштабирования дуг
def calculate_body_metrics(landmarks, visibility_threshold=0.5):
    head = np.array([landmarks[0].x, landmarks[0].y]) if landmarks[0].visibility > visibility_threshold else None
    left_heel = np.array([landmarks[31].x, landmarks[31].y]) if landmarks[31].visibility > visibility_threshold else None
    right_heel = np.array([landmarks[32].x, landmarks[32].y]) if landmarks[32].visibility > visibility_threshold else None
    height = 1.0
    if head is not None and left_heel is not None and right_heel is not None:
        feet_center = (left_heel + right_heel) / 2
        height = np.linalg.norm(head - feet_center)
    elif head is not None and landmarks[23].visibility > visibility_threshold and landmarks[24].visibility > visibility_threshold:
        pelvis_center = (np.array([landmarks[23].x, landmarks[23].y]) + np.array([landmarks[24].x, landmarks[24].y])) / 2
        height = np.linalg.norm(head - pelvis_center) * 2
    elif landmarks[11].visibility > visibility_threshold and landmarks[12].visibility > visibility_threshold:
        shoulder_center = (np.array([landmarks[11].x, landmarks[11].y]) + np.array([landmarks[12].x, landmarks[12].y])) / 2
        pelvis_center = (np.array([landmarks[23].x, landmarks[23].y]) + np.array([landmarks[24].x, landmarks[24].y])) / 2 if landmarks[23].visibility > visibility_threshold and landmarks[24].visibility > visibility_threshold else shoulder_center
        height = np.linalg.norm(shoulder_center - pelvis_center) * 3
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y]) if landmarks[11].visibility > visibility_threshold else None
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y]) if landmarks[12].visibility > visibility_threshold else None
    width = 1.0
    if left_shoulder is not None and right_shoulder is not None:
        width = np.linalg.norm(left_shoulder - right_shoulder)
    return height, width

# Список троек точек для анализа
angle_triplets = [
    (20, 16, 14),  # Правая кисть
    (16, 14, 12),  # Правый локоть
    (14, 12, 24),  # Правое плечо-таз
    (12, 24, 26),  # Правое бедро
    (24, 26, 28),  # Правое колено
    (26, 28, 32),  # Правая лодыжка
    (19, 15, 13),  # Левая кисть
    (15, 13, 11),  # Левый локоть
    (13, 11, 23),  # Левое плечо-таз
    (11, 23, 25),  # Левое бедро
    (23, 25, 27),  # Левое колено
    (25, 27, 31),  # Левая лодыжка
]

# Ожидаемые столбцы в CSV
expected_columns = ['frame', 'time', 'height', 'width'] + [f'angle_{a}_{b}_{c}' for a, b, c in angle_triplets]

# Базовая директория
input_dir = 'C:/Moution_Health/mediapipe_project'

# Получение списка директорий
video_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
print(f"Доступные папки: {video_dirs}")

# Запрос выбора папки у пользователя
if not video_dirs:
    print("Ошибка: В директории нет папок с видео и CSV")
    exit()
video_dir = input("Введите имя папки (например, shoulders): ").strip()
if not video_dir in video_dirs:
    print(f"Ошибка: Папка '{video_dir}' не найдена в {input_dir}")
    exit()

# Проверка видео и CSV
video_path = None
for ext in ['.mp4', '.mov']:
    potential_path = os.path.join(input_dir, video_dir, f"{video_dir}{ext}")
    if os.path.exists(potential_path):
        video_path = potential_path
        break
csv_path = os.path.join(input_dir, video_dir, f"{video_dir}_angles.csv")

# Проверка существования видео и CSV
print(f"\nПроверка {video_dir}:")
if not video_path or not os.path.exists(video_path):
    print(f"  Видео для {video_dir} не найдено, завершение")
    exit()
print(f"  Видео найдено: {video_path}")
if not os.path.exists(csv_path):
    print(f"  CSV-файл для {video_dir} не найден, завершение")
    exit()
print(f"  CSV найден: {csv_path}")

# Чтение CSV-файла
csv_data = []
try:
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # Проверка наличия всех ожидаемых столбцов
        if not all(col in reader.fieldnames for col in expected_columns):
            print(f"  Ошибка: CSV-файл {csv_path} не содержит всех ожидаемых столбцов {expected_columns}")
            exit()
        for row in reader:
            csv_data.append({k: float(v) if k != 'frame' else int(v) for k, v in row.items()})
        print(f"  Загружено {len(csv_data)} строк из CSV")
except Exception as e:
    print(f"  Ошибка при чтении CSV-файла {csv_path}: {e}")
    exit()

# Инициализация видео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"  Ошибка: Не удалось открыть видео {video_path}")
    exit()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  Видео содержит {total_frames} кадров")

# Настройка окна
cv2.namedWindow('CSV Verification', cv2.WINDOW_NORMAL)
frame_count = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print(f"  Завершено воспроизведение {video_path}")
            break
        frame_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            print(f"  Завершено воспроизведение {video_path}")
            break

    # Отзеркаливание кадра
    frame = cv2.flip(frame, 1)

    # Конвертация в RGB для MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Получение углов из CSV для текущего кадра
    csv_row = None
    for row in csv_data:
        if row['frame'] == frame_count - 1:  # frame_count начинается с 1, CSV с 0
            csv_row = row
            break

    # Отрисовка дуг и углов
    if results.pose_landmarks and csv_row:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        height, width = calculate_body_metrics(landmarks, visibility_threshold=0.5)

        # Отладочный вывод
        print(f"Frame {frame_count}:")
        for a, b, c in angle_triplets:
            angle_key = f'angle_{a}_{b}_{c}'
            angle = csv_row.get(angle_key, 0)
            visible = (landmarks[a].visibility > 0.5 and
                       landmarks[b].visibility > 0.5 and
                       landmarks[c].visibility > 0.5)
            print(f"  {angle_key}: Angle={angle:.1f}°, Visible={visible}")

            if angle > 0.1 and visible:
                point_a = [landmarks[a].x, landmarks[a].y]
                point_b = [landmarks[b].x, landmarks[b].y]
                point_c = [landmarks[c].x, landmarks[c].y]
                draw_arc(image, point_a, point_b, point_c, angle, arc_radius, w, h, arc_color)
                center = (int(point_b[0] * w), int(point_b[1] * h))
                text_pos = (center[0] + 10, center[1] + 10)
                image = put_text_pil(image, f"CSV: {angle:.1f}°", text_pos, font_size, text_color)

    # Предупреждение, если данные отсутствуют
    if not csv_row:
        image = put_text_pil(image, f"No CSV data for frame {frame_count}", (10, 90), font_size=20, color=warning_color)

    # Отображение номера кадра и статуса
    image = put_text_pil(image, f"Frame: {frame_count}", (10, 30), font_size=20, color=status_color)
    status_text = "Paused" if paused else "Playing"
    image = put_text_pil(image, status_text, (10, 60), font_size=20, color=status_color)

    # Отображение видео
    cv2.imshow('CSV Verification', image)

    # Задержка для стабильности
    key = cv2.waitKey(33 if not paused else 0) & 0xFF
    if key == ord('q'):
        print(f"  Прервано пользователем для {video_path}")
        break
    elif key == ord(' '):
        paused = not paused
    elif paused and key == ord('a'):
        frame_count = max(1, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    elif paused and key == ord('d'):
        frame_count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)

cap.release()
cv2.destroyAllWindows()

# Освобождение ресурсов
pose.close()
print("Обработка завершена")
