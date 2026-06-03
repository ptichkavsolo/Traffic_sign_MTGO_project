# Traffic_sign_MTGO_project
# Классификация дорожных знаков

Система для автоматического распознавания дорожных знаков на изображениях на основе свёрточной нейронной сети (CNN).

## Описание

Система принимает на вход фотографию дорожной сцены, обнаруживает знаки с помощью цветового детектора и классифицирует их по 155 классам дорожных знаков РФ. Точность классификации — 98% на валидационной выборке.

## Технологии

- Python 3.x
- Django
- TensorFlow / Keras
- OpenCV
- Pillow

## Установка и запуск

1. Клонируй репозиторий:
git clone https://github.com/ptichkavsolo/Traffic_sign_MTGO_project.git
cd Traffic_sign_MTGO_project

2. Установи зависимости:
pip install -r requirements.txt

3. Запусти сервер:
python manage.py runserver

4. Открой в браузере:
http://127.0.0.1:8000

## Датасет

Модель обучена на датасете RTSD (Russian Traffic Sign Dataset):
- 54 188 изображений дорожных сцен
- 95 492 размеченных знака
- 155 классов по ПДД РФ

Источник: https://www.kaggle.com/datasets/watchman/rtsd-dataset
