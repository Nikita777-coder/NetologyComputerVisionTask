# Подсчет автомобилей на видео

Решение для задания: YOLOv8 детектирует автомобили класса `car` из COCO, BoT-SORT сохраняет ID объектов между кадрами, а скрипт считает пересечение заданной линии только в выбранном направлении.

## Возможности

- Детекция автомобилей через Ultralytics YOLOv8.
- Трекинг через встроенный `botsort.yaml`.
- Отрисовка bounding box, ID трека, класса и центра объекта.
- Отрисовка линии подсчета.
- Подсчет пересечений без повторного учета одного и того же `track_id`.
- CSV-отчет с кадром, временем, ID объекта и координатой центра.
- Опциональное распознавание номеров, если есть отдельная YOLO-модель для номерных знаков.

## Подготовка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для режима распознавания номеров:

```bash
pip install -r requirements-plates.txt
```

## Видео для тестирования

Подойдет любое видео, где хорошо видны проезжающие автомобили. Например, можно скачать бесплатный ролик с Pexels:

- https://www.pexels.com/search/videos/traffic/

Сохраните файл, например, как:

```text
input/traffic.mp4
```

## Запуск

Пример для вертикальной линии на видео 1280x720: считаются машины, которые пересекают линию слева направо.

```bash
python count_cars.py \
  --input input/traffic.mp4 \
  --output output/output_counted.mp4 \
  --csv output/crossings.csv \
  --line-start 640,0 \
  --line-end 640,720 \
  --direction left-to-right
```

Пример для горизонтальной линии: считаются машины, которые едут сверху вниз.

```bash
python count_cars.py \
  --input input/traffic.mp4 \
  --line-start 0,420 \
  --line-end 1280,420 \
  --direction top-to-bottom
```

Если нужно увидеть обработку в реальном времени, добавьте `--show`. Нажмите `q`, чтобы закрыть окно.

## Как подобрать линию

Координаты задаются в пикселях исходного видео:

- `--line-start x1,y1` - первая точка линии.
- `--line-end x2,y2` - вторая точка линии.

Для видео 1280x720:

- вертикальная линия по центру: `--line-start 640,0 --line-end 640,720`;
- горизонтальная линия ниже середины: `--line-start 0,420 --line-end 1280,420`.

## Распознавание номеров

Базовая YOLOv8-модель `yolov8n.pt` не распознает номера, поэтому добавлена отдельная готовая модель:

```text
models/license_plate_detector.pt
```

Источник готовой модели: `Koushim/yolov8-license-plate-detection` на Hugging Face.

После установки дополнительных зависимостей запустите:

```bash
python count_cars.py \
  --input input/traffic.mp4 \
  --output output/output_with_plates.mp4 \
  --csv output/crossings_with_plates.csv \
  --line-start 0,420 \
  --line-end 1280,420 \
  --direction top-to-bottom \
  --enable-plates
```

Если номер распознан, он появится на bounding box и в колонке `plate_text` файла `crossings.csv`.

## Результаты

После выполнения появятся:

- `output/output_counted.mp4` - видео с визуализацией;
- `output/crossings.csv` - отчет по пересечениям.

CSV содержит:

```text
frame,time_seconds,track_id,class_name,center_x,center_y,plate_text
```
# NetologyComputerVisionTask
