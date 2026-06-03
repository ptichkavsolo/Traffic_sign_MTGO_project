import os
import cv2
import pandas as pd
import random
from tqdm import tqdm

# Настройки
CSV_PATH = r"D:\OneDrive\Desktop\mtmo_project\traffic_sign_project\dataset\full-gt.csv"
FRAMES_DIR = r"D:\OneDrive\Desktop\mtmo_project\traffic_sign_project\dataset\rtsd-frames"
OUTPUT_DIR = r"D:\OneDrive\Desktop\mtmo_project\traffic_sign_project\dataset\data"

IMG_SIZE = 128          # размер изображения для модели
TRAIN_RATIO = 0.7       # доля обучающей выборки
VALIDATION_RATIO = 0.1  # доля валидационной
TEST_RATIO = 0.2        # доля тестовой
MIN_SIGN_SIZE = 10      # минимальный размер знака в пикселях

random.seed(42)

# Создание папок
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Чтение CSV
df = pd.read_csv(CSV_PATH)
print(f"Всего аннотаций в CSV: {len(df)}")

# Счетчики
saved_count = 0
skipped_count = 0
class_counter = {}

# Основной цикл
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["filename"]
    sign_class = str(row["sign_class"])

    x = int(row["x_from"])
    y = int(row["y_from"])
    w = int(row["width"])
    h = int(row["height"])

    # Пропускаем слишком маленькие знаки
    if w < MIN_SIGN_SIZE or h < MIN_SIGN_SIZE:
        skipped_count += 1
        continue

    img_path = os.path.join(FRAMES_DIR, filename)
    if not os.path.exists(img_path):
        skipped_count += 1
        continue

    # Читаем изображение
    img = cv2.imread(img_path)
    if img is None:
        skipped_count += 1
        continue

    img_h, img_w = img.shape[:2]

    # Защита от выхода за границы
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    crop = img[y:y2, x:x2]

    if crop.size == 0:
        skipped_count += 1
        continue

    # Resize до IMG_SIZE x IMG_SIZE
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    # Выбираем, в какую папку положить
    r = random.random()
    if r < TRAIN_RATIO:
        split = "train"
    elif r < TRAIN_RATIO + VALIDATION_RATIO:
        split = "validation"
    else:
        split = "test"

    # Создаем папку класса
    class_dir = os.path.join(OUTPUT_DIR, split, sign_class)
    os.makedirs(class_dir, exist_ok=True)

    # Уникальное имя файла
    out_name = f"{sign_class}_{idx}.jpg"
    out_path = os.path.join(class_dir, out_name)

    # Сохраняем
    cv2.imwrite(out_path, crop)
    saved_count += 1
    class_counter[sign_class] = class_counter.get(sign_class, 0) + 1

# Итог
print("\nГотово")
print(f"Сохранено изображений: {saved_count}")
print(f"Пропущено аннотаций: {skipped_count}")
print(f"Количество классов: {len(class_counter)}")
