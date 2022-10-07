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

Inference: https://drive.google.com/file/d/1ZxBmfltLspRGKSkNJ7tt2epph4MpcwiM/view?usp=sharing
