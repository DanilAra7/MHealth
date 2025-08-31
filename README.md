# Motion Health Project

Проект для анализа движений с использованием MediaPipe. Позволяет обрабатывать видео, вычислять углы суставов, сохранять их в CSV и проверять CSV-файлы с визуализацией.

## Установка

1. Клонируйте репозиторий:

git clone https://github.com/your-username/motion-health.git
cd motion-health

2. Создайте и активируйте виртуальное окружение:
python -m venv mediapipe_env
.\mediapipe_env\Scripts\activate  # Windows
source mediapipe_env/bin/activate  # Linux/Mac

3. Установите зависимости:
pip install -r requirements.txt

4. Подготовьте видео:
    В папку videos загрузите видеозаписи выполнения упражнений

5. Запустите record_angles