# Car plate recognition

Vehicle registration plates are saved in the "logs" folder

## Setup
```Linux Kernel Module
conda create -n env_name --python=3.10

conda activate env_name

pip install -r requirements.txt
``` 
## Usage
#### Detection on:
1. Image
```Linux Kernel Module
python main.py --weights models/y5m6.pt --source data/sample.jpg --img 1280
``` 
2. Video
```Linux Kernel Module
python main.py --weights models/y5s_baseline.pt --source data/sample2.mp4
```
3. Test dataset
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
![image](https://user-images.githubusercontent.com/51293938/197833647-219bad18-dca4-4486-b711-189354bb688f.png)

## Тренировочные параметры
• Batch size (размер батча, то есть количество картинок, одновременно подаваемых на вход yolo) : 16

• Image size (размер изображения, подаваемого на вход yolo. Это значит, что размер исходного изображения преобразуется к виду n x n, где n — число, введенное пользователем после ключа —img. ВАЖНО: n должно быть кратно 32 - это связано с архитектурой yolo): 640x640

• Epochs (количество эпох для обучения): 100
Model	Precision	Recall	mAP@[.5]	mAP@[.5:.95]
yolov7-e6	0.2246	0.3684	0.1329	0.04663
yolov7-w6	0.9885	0.05263	0.05659	0.005891
yolov7x	0.2692	0.2632	0.1553	0.04348
yolov7	0.9999	1	0.9952	0.7518
![image](https://user-images.githubusercontent.com/110126453/197833730-1293a7b1-0437-415c-9311-87fb642fab02.png)

Inference: https://drive.google.com/file/d/1ZxBmfltLspRGKSkNJ7tt2epph4MpcwiM/view?usp=sharing
