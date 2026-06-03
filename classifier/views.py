import os
import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from .class_labels import CLASS_LABELS
import json

# Загрузка модели и классов
MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier/model/best_model.h5')
model = load_model(MODEL_PATH)

CLASS_NAMES_PATH = os.path.join(settings.BASE_DIR, 'classifier/model/class_names.json')
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = json.load(f)

def home(request):
    return render(request, 'classifier/index.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        img_path = os.path.join(settings.MEDIA_ROOT, img_file.name)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(img_path, 'wb+') as f:
            for chunk in img_file.chunks():
                f.write(chunk)
        return JsonResponse({'image_name': img_file.name})
    return JsonResponse({'error': 'нет файла'}, status=400)

# Детекция знаков
def detect_traffic_signs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Диапазоны цветов знаков
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
           cv2.inRange(hsv, lower_red2, upper_red2) | \
           cv2.inRange(hsv, lower_blue, upper_blue) | \
           cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:  # фильтр мелких объектов
            continue
        rects.append([x, y, x+w, y+h])

    if len(rects) == 0:
        return None

    # Объединяем перекрывающиеся прямоугольники
    def merge_rects(rects):
        merged = []
        taken = [False]*len(rects)
        for i in range(len(rects)):
            if taken[i]:
                continue
            x1, y1, x2, y2 = rects[i]
            for j in range(i+1, len(rects)):
                if taken[j]:
                    continue
                xx1, yy1, xx2, yy2 = rects[j]
                if x1 < xx2 and x2 > xx1 and y1 < yy2 and y2 > yy1:
                    x1 = min(x1, xx1)
                    y1 = min(y1, yy1)
                    x2 = max(x2, xx2)
                    y2 = max(y2, yy2)
                    taken[j] = True
            merged.append([x1, y1, x2, y2])
        return merged
    
    merged_rects = merge_rects(rects)
    results = []
    for x1, y1, x2, y2 in merged_rects:
        pad = int(0.1 * max(x2-x1, y2-y1))
        x1_pad = max(0, x1-pad)
        y1_pad = max(0, y1-pad)
        x2_pad = min(img.shape[1], x2+pad)
        y2_pad = min(img.shape[0], y2+pad)
        crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
        results.append((crop, (x1_pad, y1_pad, x2_pad-x1_pad, y2_pad-y1_pad)))
    return results

# Предсказание
def predict_image(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'не POST'}, status=400)

    img_name = request.POST.get('image_name')
    if not img_name:
        return JsonResponse({'error': 'не указано имя изображения'}, status=400)

    img_path = os.path.join(settings.MEDIA_ROOT, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return JsonResponse({'error': 'не удалось открыть изображение'}, status=400)

    detections = detect_traffic_signs(img)
    
    # Если знаков не найдено
    if detections is None:
        return JsonResponse({
            'found': False,
            'message': 'Дорожные знаки не найдены на изображении'
        })
    
    if len(detections) == 0:
        return JsonResponse({
            'found': False,
            'message': 'Дорожные знаки не найдены на изображении'
        })

    results = []
    for crop, (x, y, w, h) in detections:
        crop_resized = cv2.resize(crop, (64,64))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        x_input = image.img_to_array(crop_rgb)/255.0
        x_input = np.expand_dims(x_input, axis=0)

        preds = model.predict(x_input, verbose=0)
        class_index = int(np.argmax(preds[0]))
        class_code = CLASS_NAMES[class_index]
        class_name = CLASS_LABELS.get(class_code, class_code)
        results.append({'bbox':[int(x),int(y),int(w),int(h)], 'class_name':class_name})

    img_out = img.copy()
    for r in results:
        x, y, w, h = r['bbox']
        cv2.rectangle(img_out, (x,y), (x+w, y+h), (0,255,0), 2)
    out_path = os.path.join(settings.MEDIA_ROOT, 'result_'+img_name)
    cv2.imwrite(out_path, img_out)

    return JsonResponse({
        'found': True,
        'results': results, 
        'result_image': 'result_' + img_name
    })