# Car plate recognition

Автомобильные номера сохраняются в папку "logs".

## Установка
```Linux Kernel Module
conda create -n env_name --python=3.10

conda activate env_name

pip install -r requirements.txt
``` 
## Применение
#### Detection on:
1. Изображение
```Linux Kernel Module
python main.py --weights models/y5m6.pt --source data/sample.jpg --img 1280
``` 
2. Видео
```Linux Kernel Module
python main.py --weights models/y5s_baseline.pt --source data/sample2.mp4
```
3. Тестовый датасет
```Linux Kernel Module
python main.py --weights models/y5m_baseline.pt --source test
```
## Устройство системы
![Untitled (2)](https://user-images.githubusercontent.com/110126453/197808757-283a6c0e-d609-41a4-8fbf-b948cb434525.jpg)
![Untitled (4)](https://user-images.githubusercontent.com/110126453/197826538-2fa5d2fc-59a8-4e2b-8309-f3728c3e011f.jpg)

#### *Желательный размер входного изображения: 1280x720px.

## Метрики
•	Accuracy – это показатель, который описывает общую точность предсказания модели по всем классам. Рассчитывается как отношение количества правильных прогнозов к их общему количеству.

•	IoU (Intersection Over Union) – используется для определения того, правильно ли была предсказана ограничивающая рамка. Соотношение перекрытия между областями двух ограничивающих прямоугольников становится равным 1.0 в случае точного совпадения и 0.0, если перекрытия нет.

•	mAP (mean average precision) – представляет собой среднее значений AP. 
AP - это среднее значение по нескольким IoU (минимальное значение IoU для рассмотрения положительного совпадения). AP@[0.5:0.95] соответствует среднему значению AP для IoU от 0,5 до 0,95 с шагом 0,05. <br/>


## Тренировочные параметры
• Batch size (размер батча, то есть количество картинок, одновременно подаваемых на вход yolo) : 16

• Image size (размер изображения, подаваемого на вход yolo. Это значит, что размер исходного изображения преобразуется к виду n x n, где n — число, введенное пользователем после ключа —img. ВАЖНО: n должно быть кратно 32 - это связано с архитектурой yolo): 640x640

• Epochs (количество эпох для обучения): 100

## Результат работы
• Детекция номера
| model  | mAP50 | speed |
| ------------- | ------------- | ------------- |
| yolov5n  | 0.86  | 10ms  |
| yolov5s  | 0.9  | 15ms  |
| yolov5s6  | 0.9  | 30ms  |
| yolov5m  | 0.92  | 20ms  |
| yolov7e6  | 0.82  | 35ms  |

• Распознавание номера

| model  | accuracy | speed |
| ------------- | ------------- | ------------- |
| easyOCR  | 0.35  | 20ms  |

• Всё решение <br/>
Accuracy всего решения - 0.2

Inference: https://drive.google.com/file/d/1ZxBmfltLspRGKSkNJ7tt2epph4MpcwiM/view?usp=sharing

## Возможность масштабирования
• Есть возможность масштабировать систему до двух и более камер при уменьшении количества кадров в секунду.

![image_2022-10-25_20-12-07](https://user-images.githubusercontent.com/110126453/197839057-b45cf107-5cff-4456-8a4d-daee9ed348bc.png)

