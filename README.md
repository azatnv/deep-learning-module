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
## Метрики
•	Accuracy – это показатель, который описывает общую точность предсказания модели по всем классам. Рассчитывается как отношение количества правильных прогнозов к их общему количеству.

•	IoU (Intersection Over Union) – используется для определения того, правильно ли была предсказана ограничивающая рамка. Соотношение перекрытия между областями двух ограничивающих прямоугольников становится равным 1.0 в случае точного совпадения и 0.0, если перекрытия нет.

•	mAP (mean average precision) – представляет собой среднее значений AP. 
AP - это среднее значение по нескольким IoU (минимальное значение IoU для рассмотрения положительного совпадения). AP@[0.5:0.95] соответствует среднему значению AP для IoU от 0,5 до 0,95 с шагом 0,05.

## Устройство системы и основные требования
![Untitled](https://user-images.githubusercontent.com/110126453/197792574-38aadb3b-4876-4d65-a599-705f17d5c60b.jpg)
#### Требования:
• Камера видеонаблюдения, выдающая изображение c минимальным размером 640x640px.

Inference: https://drive.google.com/file/d/1ZxBmfltLspRGKSkNJ7tt2epph4MpcwiM/view?usp=sharing
