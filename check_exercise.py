import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from math import acos, degrees
from PIL import Image, ImageDraw, ImageFont
import os

# Параметры точности и визуализации
angle_variance_threshold = 50.0
angle_error_threshold = 15.0
static_angle_tolerance = 5.0
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
lookahead_frames = 15
user_history_frames = 30
arc_thickness = 6
arc_color = (0, 255, 255)
error_arc_color = (0, 0, 255)
arc_radius = 30
MAX_FRAME_JUMP = 40  # Ограничение на максимальный скачок closest_frame

# Функция для вычисления угла между тремя точками с нормализацией
def calculate_angle(a, b, c, height, real_height=1.8):
    if height is None or height == 0:
        return None
    scale = real_height / height
    a_scaled = np.array([a.x * scale, a.y * scale])
    b_scaled = np.array([b.x * scale, b.y * scale])
    c_scaled = np.array([c.x * scale, c.y * scale])
    
    ba = a_scaled - b_scaled
    bc = c_scaled - b_scaled
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Функция для наложения текста с поддержкой кириллицы
def put_text_pil(image, text, position, font_size=20, color=(0, 0, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line:
            draw.text((position[0], position[1] + i * (font_size + 10)), line, font=font, fill=color)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image

# Функция для вычисления углов поворота векторов
def calculate_vector_angle(point1, point2):
    vector = np.array(point2) - np.array(point1)
    angle = degrees(np.arctan2(vector[1], vector[0]))
    return angle

# Функция для вычисления нормализованных координат для углов
def calculate_body_metrics(landmarks, visibility_threshold=0.5):
    head = np.array([landmarks[0].x, landmarks[0].y]) if landmarks[0].visibility > visibility_threshold else None
    left_heel = np.array([landmarks[31].x, landmarks[31].y]) if landmarks[31].visibility > visibility_threshold else None
    right_heel = np.array([landmarks[32].x, landmarks[32].y]) if landmarks[32].visibility > visibility_threshold else None
    height = None
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

# Функция для отрисовки дуги
def draw_arc(image, point_a, point_b, point_c, angle_curr, angle_target, radius, width, height, color):
    center = (int(point_b[0] * width), int(point_b[1] * height))
    angle_ab = calculate_vector_angle(point_b, point_a)
    angle_bc = calculate_vector_angle(point_b, point_c)
    angle_ab = (angle_ab + 360) % 360
    angle_bc = (angle_bc + 360) % 360
    smaller = min(angle_ab, angle_bc)
    larger = max(angle_ab, angle_bc)
    short_diff = larger - smaller
    if short_diff > 180:
        smaller, larger = larger, smaller
        short_diff = 360 - short_diff
    if angle_curr < angle_target:
        start_angle = larger
        end_angle = smaller + 360
    else:
        start_angle = smaller
        end_angle = larger
    arc_measure = end_angle - start_angle
    cv2.ellipse(image, center, (int(radius / max(width, height) * width), int(radius / max(width, height) * height)),
                0, start_angle, end_angle, color, arc_thickness)

# Функция для анализа тенденции угла
def analyze_angle_trend(angles, start_frame, end_frame, prev_trend=None):
    if start_frame >= len(angles) or end_frame > len(angles) or end_frame <= start_frame:
        return prev_trend, None, None
    segment = angles[start_frame:end_frame]
    if len(segment) < 2:
        return prev_trend, None, None
    diffs = np.diff(segment)
    increasing = np.all(diffs >= -2.0)
    decreasing = np.all(diffs <= 2.0)
    if increasing and not decreasing:
        return 'increasing', min(segment), max(segment)
    elif decreasing and not increasing:
        return 'decreasing', min(segment), max(segment)
    else:
        max_idx = np.argmax(segment) + start_frame
        min_idx = np.argmin(segment) + start_frame
        if max_idx > start_frame and max_idx < end_frame - 1:
            return 'decreasing' if prev_trend != 'decreasing' else prev_trend, min(segment), max(segment)
        elif min_idx > start_frame and min_idx < end_frame - 1:
            return 'increasing' if prev_trend != 'increasing' else prev_trend, min(segment), max(segment)
        return prev_trend, min(segment), max(segment)

# Маппинг номеров точек на анатомические названия
landmark_names = {
    0: "нос", 1: "левый глаз (внутренний)", 2: "левый глаз", 3: "левый глаз (внешний)",
    4: "правый глаз (внутренний)", 5: "правый глаз", 6: "правый глаз (внешний)",
    7: "левое ухо", 8: "правое ухо", 9: "левый угол рта", 10: "правый угол рта",
    11: "левое плечо", 12: "правое плечо", 13: "левый локоть", 14: "правый локоть",
    15: "левое запястье", 16: "правое запястье", 17: "левый мизинец", 18: "правый мизинец",
    19: "левый указательный палец", 20: "правый указательный палец", 21: "левый большой палец",
    22: "правый большой палец", 23: "левый таз", 24: "правый таз", 25: "левое бедро",
    26: "правое бедро", 27: "левое колено", 28: "правое колено", 29: "левая лодыжка",
    30: "правая лодыжка", 31: "левая пятка", 32: "правая пятка"
}

# Список троек точек для анализа
joints = [
    {'name': 'правая кисть', 'triple': (20, 16, 14)},
    {'name': 'правый локоть', 'triple': (16, 14, 12)},
    {'name': 'правое плечо-таз', 'triple': (14, 12, 24)},
    {'name': 'правое бедро', 'triple': (12, 24, 26)},
    {'name': 'правое колено', 'triple': (24, 26, 28)},
    {'name': 'правая лодыжка', 'triple': (26, 28, 32)},
    {'name': 'левая кисть', 'triple': (19, 15, 13)},
    {'name': 'левый локоть', 'triple': (15, 13, 11)},
    {'name': 'левое плечо-таз', 'triple': (13, 11, 23)},
    {'name': 'левое бедро', 'triple': (11, 23, 25)},
    {'name': 'левое колено', 'triple': (23, 25, 27)},
    {'name': 'левая лодыжка', 'triple': (25, 27, 31)},
]

# Выбор упражнения
exercise = "shoulders"
folder = exercise
csv_path = os.path.join(folder, f"{exercise}_angles.csv")
ref_video_path = os.path.join(folder, f"{exercise}.mov")

# Загрузка эталонных данных
reference_data = pd.read_csv(csv_path)

# Определение активных и неподвижных суставов
active_joints = []
static_joints = []
ref_angles = {}
global_min_max = {}
for j, joint in enumerate(joints):
    angle_col = f"angle_{joint['triple'][0]}_{joint['triple'][1]}_{joint['triple'][2]}"
    angles = reference_data[angle_col].values
    ref_angles[j] = angles
    valid_angles = angles[angles != 0.0]
    if len(valid_angles) == 0 or np.any(np.isnan(valid_angles)):
        static_joints.append(j)
        print(f"Сустав {joint['name']} неподвижен: все углы 0.0 или содержат NaN")
    else:
        angle_range = np.max(valid_angles) - np.min(valid_angles)
        if angle_range > angle_variance_threshold:
            active_joints.append(j)
            global_min_max[j] = (np.min(valid_angles), np.max(valid_angles))
            print(f"Сустав {joint['name']} активен: диапазон={angle_range:.1f}")
        else:
            static_joints.append(j)
            print(f"Сустав {joint['name']} неподвижен: диапазон={angle_range:.1f}")

print(f"Активные суставы для {exercise}: {[joints[j]['name'] for j in active_joints]}")
print(f"Неподвижные суставы для {exercise}: {[joints[j]['name'] for j in static_joints]}")

# Предварительное вычисление тенденций для эталонного видео
ref_trends = {}
prev_trends = {j: None for j in active_joints}  # Храним предыдущие тенденции для эталона
for frame_idx in range(len(reference_data)):
    ref_trends[frame_idx] = {}
    for j in active_joints:
        trend, min_angle, max_angle = analyze_angle_trend(
            ref_angles[j], frame_idx, min(frame_idx + lookahead_frames, len(reference_data)), prev_trends[j]
        )
        ref_trends[frame_idx][j] = trend
        prev_trends[j] = trend  # Обновляем предыдущую тенденцию

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)

# Ввод реального роста пользователя
real_height = 1.72

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Загрузка эталонного видео
ref_cap = cv2.VideoCapture(ref_video_path)
if not ref_cap.isOpened():
    print("Ошибка: Не удалось загрузить эталонное видео")
    ref_cap = None

# Настройка окна
cv2.namedWindow('Motion Feedback', cv2.WINDOW_NORMAL)
fullscreen = False

# Переменные для отслеживания
last_update_frame = -lookahead_frames
frame_count = 0
current_trends = [None] * len(joints)
angle_ranges = [(None, None)] * len(joints)
ref_start_frame = 0
technique_status = ""
user_angle_history = {j: [] for j in active_joints}
trend_error = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Отзеркаливание видео пользователя
    frame = cv2.flip(frame, 1)

    # Конвертация в RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Обработка изображения
    results = pose.process(image)

    # Конвертация обратно в BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    has_errors = False
    trend_error = ""

    # Отображение эталонного видео
    if ref_cap:
        ret_ref, ref_frame = ref_cap.read()
        if not ret_ref:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ref, ref_frame = ref_cap.read()
        if ret_ref:
            ref_frame = cv2.flip(ref_frame, 1)
            ref_h, ref_w = ref_frame.shape[:2]
            target_w, target_h = 160, 120
            scale = min(target_w / ref_w, target_h / ref_h)
            new_w, new_h = int(ref_w * scale), int(ref_h * scale)
            ref_frame = cv2.resize(ref_frame, (new_w, new_h))
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            canvas[top:top+new_h, left:left+new_w] = ref_frame
            image[0:target_h, 0:target_w] = canvas

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        height, width = calculate_body_metrics(landmarks, min_tracking_confidence)

        # Вычисление текущих углов с нормализацией
        current_angles = []
        joint_points = []
        for j, joint in enumerate(joints):
            triple = joint['triple']
            if (landmarks[triple[0]].visibility > min_tracking_confidence and
                landmarks[triple[1]].visibility > min_tracking_confidence and
                landmarks[triple[2]].visibility > min_tracking_confidence):
                curr_angle = calculate_angle(landmarks[triple[0]], landmarks[triple[1]], landmarks[triple[2]], height, real_height)
                points = (
                    [landmarks[triple[0]].x, landmarks[triple[0]].y],
                    [landmarks[triple[1]].x, landmarks[triple[1]].y],
                    [landmarks[triple[2]].x, landmarks[triple[2]].y]
                )
            else:
                curr_angle = None
                points = None
            current_angles.append(curr_angle)
            joint_points.append(points)

        # Обновление истории углов пользователя
        for j in active_joints:
            if current_angles[j] is not None:
                user_angle_history[j].append(current_angles[j])
                if len(user_angle_history[j]) > user_history_frames:
                    user_angle_history[j].pop(0)

        # Вычисление тенденций пользователя
        user_trends = {}
        for j in active_joints:
            if len(user_angle_history[j]) >= user_history_frames:
                avg_angle = np.mean(user_angle_history[j])
                if current_angles[j] is not None:
                    if current_angles[j] < avg_angle:
                        user_trends[j] = 'decreasing'
                    elif current_angles[j] > avg_angle:
                        user_trends[j] = 'increasing'
                    else:
                        user_trends[j] = None
            else:
                user_trends[j] = None

        # Выбор ближайшего кадра с учётом совпадения тенденций
        if frame_count - last_update_frame >= lookahead_frames or frame_count == 0:
            if ref_start_frame >= len(reference_data):
                print(f"Достигнут конец эталонного видео, сброс ref_start_frame на 0")
                ref_start_frame = 0
            reference_data_sliced = reference_data.iloc[ref_start_frame:]
            valid_indices = [i for i, angle in enumerate(current_angles) if angle is not None and i in active_joints]
            if valid_indices:
                min_distance = float('inf')
                closest_frame = ref_start_frame
                trend_match_found = False
                # Ограничиваем поиск до ref_start_frame + MAX_FRAME_JUMP
                max_frame = min(len(reference_data_sliced), MAX_FRAME_JUMP)
                for frame_idx in range(max_frame):
                    global_frame_idx = frame_idx + ref_start_frame
                    trends_match = all(
                        user_trends.get(j) == ref_trends.get(global_frame_idx, {}).get(j)
                        for j in valid_indices
                    )
                    if trends_match:
                        distance = sum(
                            (ref_angles[j][global_frame_idx] - current_angles[j]) ** 2
                            for j in valid_indices
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_frame = global_frame_idx
                            trend_match_found = True
                if not trend_match_found:
                    trend_error = "Тенденция не совпадает!"
                    closest_frame = ref_start_frame

                # Обновление текущих тенденций для отображения
                for j in active_joints:
                    new_trend = ref_trends.get(closest_frame, {}).get(j)
                    if new_trend is None:
                        # Сохраняем предыдущую тенденцию, если новая None
                        current_trends[j] = current_trends[j] or 'increasing'
                    else:
                        current_trends[j] = new_trend
                    angle_slice = ref_angles[j][closest_frame:min(closest_frame + lookahead_frames, len(reference_data))]
                    angle_ranges[j] = (
                        min(angle_slice) if len(angle_slice) > 0 else None,
                        max(angle_slice) if len(angle_slice) > 0 else None
                    )

                # Обновление ref_start_frame для "забывания" всех кадров до closest_frame
                ref_start_frame = closest_frame + 1
                last_update_frame = frame_count
            else:
                closest_frame = ref_start_frame
                trend_error = "Тенденция не совпадает!"

        # Проверка углов и отрисовка дуг
        for j in active_joints:
            triple = joints[j]['triple']
            if (landmarks[triple[0]].visibility > min_tracking_confidence and
                landmarks[triple[1]].visibility > min_tracking_confidence and
                landmarks[triple[2]].visibility > min_tracking_confidence):
                curr_angle = current_angles[j]
                min_angle, max_angle = global_min_max[j]
                point_a, point_b, point_c = joint_points[j]
                if curr_angle is not None and (curr_angle < min_angle or curr_angle > max_angle):
                    has_errors = True
                    target_angle = max(min_angle, min(curr_angle, max_angle))
                    draw_arc(image, point_a, point_b, point_c, curr_angle, target_angle, arc_radius, w, h, error_arc_color)
                elif curr_angle is not None and current_trends[j] in ['increasing', 'decreasing']:
                    start_angle = curr_angle
                    end_angle = ref_angles[j][min(closest_frame + lookahead_frames, len(reference_data) - 1)]
                    draw_arc(image, point_a, point_b, point_c, start_angle, end_angle, arc_radius, w, h, arc_color)

        for j in static_joints:
            triple = joints[j]['triple']
            if (landmarks[triple[0]].visibility > min_tracking_confidence and
                landmarks[triple[1]].visibility > min_tracking_confidence and
                landmarks[triple[2]].visibility > min_tracking_confidence):
                curr_angle = calculate_angle(landmarks[triple[0]], landmarks[triple[1]], landmarks[triple[2]], height, real_height)
                ref_angle = ref_angles[j][min(closest_frame, len(reference_data) - 1)]
                point_a, point_b, point_c = joint_points[j]
                if curr_angle is not None and abs(curr_angle - ref_angle) > angle_error_threshold + static_angle_tolerance:
                    has_errors = True
                    draw_arc(image, point_a, point_b, point_c, curr_angle, ref_angle, arc_radius, w, h, error_arc_color)

        # Обновление статуса техники
        technique_status = "Техника правильная!" if not has_errors else ""

    # Отображение статуса техники, чекпоинта, тенденций и ошибки тенденции
    if trend_error:
        image = put_text_pil(image, trend_error, (10, 110), font_size=20, color=(0, 0, 255))
    if technique_status:
        image = put_text_pil(image, technique_status, (10, 140), font_size=20, color=(0, 255, 0))
    image = put_text_pil(image, f"Чекпоинт: {closest_frame}", (10, 170), font_size=20, color=(0, 255, 0))
    for i, j in enumerate(active_joints):
        trend_text = {
            'increasing': 'увеличивается',
            'decreasing': 'уменьшается',
            None: 'нет тенденции'
        }.get(current_trends[j], 'нет тенденции')
        image = put_text_pil(image, f"{joints[j]['name']}: {trend_text}", (10, 200 + i * 30), font_size=20, color=(0, 255, 0))

    # Отображение изображения
    cv2.imshow('Motion Feedback', image)

    # Обработка клавиш
    key = cv2.waitKey(10) & 0xFF
    if key in [ord('q'), ord('й'), 27]:
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty('Motion Feedback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Motion Feedback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    frame_count += 1

# Освобождение ресурсов
cap.release()
if ref_cap:
    ref_cap.release()
cv2.destroyAllWindows()
pose.close()