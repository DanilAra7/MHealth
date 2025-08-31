import cv2
import mediapipe as mp
import numpy as np
from math import degrees
from PIL import Image, ImageDraw, ImageFont

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# Параметры визуализации
arc_thickness = 3  # Толщина дуги
arc_color = (255, 0, 255)  # Цвет дуги (магента для видимости)
arc_radius = 30  # Радиус дуг в пикселях

# Функция для вычисления угла между тремя точками (A, B, C), где B - вершина
def calculate_angle(point_a, point_b, point_c):
    a = np.array([point_a.x, point_a.y, point_a.z])
    b = np.array([point_b.x, point_b.y, point_b.z])
    c = np.array([point_c.x, point_c.y, point_c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    print(f"Cosine: {cosine_angle}, Angle: {np.degrees(angle)}")
    return np.degrees(angle)

# Функция для вычисления углов поворота векторов для дуги
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
    if abs(angle_curr) > 0.1:  # Чтобы избежать нулевых углов
        cv2.ellipse(image, center, (int(radius / max(width, height) * width), int(radius / max(width, height) * height)),
                    0, start_angle, end_angle, color, arc_thickness)
    return start_angle, end_angle

# Функция для наложения текста с углами
def put_text_pil(image, text, position, font_size=15, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except IOError:
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

# Инициализация камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Камера не найдена")
    exit()

frame_count = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра с камеры")
            break
        frame_count += 1
    else:
        # В режиме паузы используем текущий кадр
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра с камеры")
            break

    # Отзеркаливание кадра
    frame = cv2.flip(frame, 1)

    # Конвертация в RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Обработка изображения
    results = pose.process(image)

    # Конвертация обратно в BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Вычисление размеров тела
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        height, width = calculate_body_metrics(landmarks, visibility_threshold=0.5)

        # Отрисовка контрольных точек и линий
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

        # Отрисовка дуг и значений углов
        for idx, (a, b, c) in enumerate(angle_triplets):
            if (landmarks[a].visibility > 0.5 and 
                landmarks[b].visibility > 0.5 and 
                landmarks[c].visibility > 0.5):
                angle = calculate_angle(landmarks[a], landmarks[b], landmarks[c])
                point_a = [landmarks[a].x, landmarks[a].y]
                point_b = [landmarks[b].x, landmarks[b].y]
                point_c = [landmarks[c].x, landmarks[c].y]
                print(f"Frame {frame_count}, {a}_{b}_{c}: A={point_a}, B={point_b}, C={point_c}, Angle={angle:.2f}")
                draw_arc(image, point_a, point_b, point_c, angle, arc_radius, w, h, arc_color)

                # Отрисовка текущего угла
                center = (int(point_b[0] * w), int(point_b[1] * h))
                text_pos = (center[0] + 10, center[1] + 10)
                text = f"Calc: {angle:.1f}°"
                image = put_text_pil(image, text, text_pos, font_size=15, color=(255, 255, 255))

    # Отображение текущего номера кадра
    image = put_text_pil(image, f"Frame: {frame_count}", (10, 30), font_size=20, color=(0, 255, 255))

    # Отображение статуса паузы
    status_text = "Paused" if paused else "Playing"
    image = put_text_pil(image, status_text, (10, 60), font_size=20, color=(0, 255, 255))

    # Отображение видео в реальном времени
    cv2.imshow('MediaPipe Pose Visualization', image)

    # Обработка клавиш
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    if key == ord('q'):  # Выход
        break
    elif key == ord(' '):  # Пауза/возобновление
        paused = not paused
    elif paused and key == ord('a'):  # Предыдущий кадр
        frame_count = max(1, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    elif paused and key == ord('d'):  # Следующий кадр
        frame_count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)

cap.release()
cv2.destroyAllWindows()

# Освобождение ресурсов
pose.close()
