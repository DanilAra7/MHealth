import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os
import shutil

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Функция для вычисления угла между тремя точками (A, B, C), где B - вершина, с нормализацией
def calculate_angle(point_a, point_b, point_c, height, real_height=1.8):
    # Масштабирование координат на основе нормализованного роста и реального роста
    scale = real_height / height
    a = np.array([point_a.x * scale, point_a.y * scale])
    b = np.array([point_b.x * scale, point_b.y * scale])
    c = np.array([point_c.x * scale, point_c.y * scale])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Функция для вычисления нормализованных координат роста и ширины
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

# Список троек точек для вычисления углов
angle_triplets = [
    # Правая сторона
    (20, 16, 14),  # Правое запястье – Правая кисть – Правый локоть
    (16, 14, 12),  # Правая кисть – Правый локоть – Правое плечо
    (14, 12, 24),  # Правый локоть – Правое плечо – Правое бедро
    (12, 24, 26),  # Правое плечо – Правое бедро – Правое колено
    (24, 26, 28),  # Правое бедро – Правое колено – Правая лодыжка
    (26, 28, 32),  # Правое колено – Правая лодыжка – Правая стопа
    # Левая сторона
    (19, 15, 13),  # Левое запястье – Левая кисть – Левый локоть
    (15, 13, 11),  # Левая кисть – Левый локоть – Левое плечо
    (13, 11, 23),  # Левый локоть – Левое плечо – Левое бедро
    (11, 23, 25),  # Левое плечо – Левое бедро – Левое колено
    (23, 25, 27),  # Левое бедро – Левое колено – Левая лодыжка
    (25, 27, 31),  # Левое колено – Левая лодыжка – Левая стопа
]

# Ввод реального роста пользователя (в метрах)
real_height = float(input("Введите ваш рост в метрах (например, 1.8): "))

# Директория с видео
input_dir = 'videos'  # Измени, если нужно
os.makedirs(input_dir, exist_ok=True)

# Список видео для обработки (поддержка .mp4 и .mov)
video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    video_name = os.path.splitext(video_file)[0]

    # Проверка существования папки
    output_folder = os.path.join('.', video_name)
    if os.path.exists(output_folder):
        print(f"Папка для {video_file} уже существует, пропускаем")
        continue

    # Создание папки для видео
    os.makedirs(output_folder, exist_ok=True)

    # Копирование оригинала видео в папку
    original_video_path = os.path.join(output_folder, video_file)
    shutil.copy(video_path, original_video_path)
    print(f"Копирование оригинала: {original_video_path}")

    # Настройка CSV
    output_file = os.path.join(output_folder, f"{video_name}_angles.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Заголовки для углов, роста и ширины
        headers = ['frame', 'time', 'height', 'width'] + [f'angle_{a}_{b}_{c}' for a, b, c in angle_triplets]
        writer.writerow(headers)

        # Захват видео
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Завершена обработка {video_file}")
                break

            # Отзеркаливание кадра
            frame = cv2.flip(frame, 1)

            # Конвертация в RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Обработка изображения
            results = pose.process(image)

            # Сохранение углов и метрик
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                height, width = calculate_body_metrics(landmarks, 0.5)
                row = [frame_count, time.time() - start_time, height, width]
                
                # Вычисление углов для каждой тройки с нормализацией
                for a, b, c in angle_triplets:
                    if (landmarks[a].visibility > 0.5 and 
                        landmarks[b].visibility > 0.5 and 
                        landmarks[c].visibility > 0.5):
                        angle = calculate_angle(landmarks[a], landmarks[b], landmarks[c], height, real_height)
                        print(f"Frame {frame_count}, {a}_{b}_{c}: Angle={angle:.2f}, Height={height:.2f}")
                    else:
                        angle = None
                    row.append(angle if angle is not None else 0)
                
                writer.writerow(row)

            frame_count += 1

        cap.release()

# Освобождение ресурсов
pose.close()