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
![Untitled (4)](https://user-images.githubusercontent.com/110126453/197826037-208ca52b-107a-47af-9d0f-1168548fb081.jpg)

#### *Желательный размер входного изображения: 1280x720px.

## Метрики
•	Accuracy – это показатель, который описывает общую точность предсказания модели по всем классам. Рассчитывается как отношение количества правильных прогнозов к их общему количеству.

•	IoU (Intersection Over Union) – используется для определения того, правильно ли была предсказана ограничивающая рамка. Соотношение перекрытия между областями двух ограничивающих прямоугольников становится равным 1.0 в случае точного совпадения и 0.0, если перекрытия нет.

•	mAP (mean average precision) – представляет собой среднее значений AP. 
AP - это среднее значение по нескольким IoU (минимальное значение IoU для рассмотрения положительного совпадения). AP@[0.5:0.95] соответствует среднему значению AP для IoU от 0,5 до 0,95 с шагом 0,05.

Inference: https://drive.google.com/file/d/1ZxBmfltLspRGKSkNJ7tt2epph4MpcwiM/view?usp=sharing
