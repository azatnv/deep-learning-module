from cv2 import VideoCapture, imwrite
import os
import csv
from pathlib import Path
from datetime import date
from uuid import uuid1
import torch

from modules.recognition.recognize import recognize_text_with_easyocr

######################################
#      Create "logs" folders to save images and csv
######################################

LOGS_FOLDER = "./logs/"
Path(LOGS_FOLDER).mkdir(exist_ok=True)


def get_today_paths():
    today = date.today().strftime("%Y-%m-%d")

    folder = Path(f"{LOGS_FOLDER}/{today}/")
    folder.mkdir(exist_ok=True)

    image_folder_path = folder / f"licenses {today}"
    image_folder_path.mkdir(exist_ok=True)

    csv_filename = folder / f"{today}.csv"

    return csv_filename, image_folder_path


CSV_PATH, IMAGE_FOLDER_PATH = get_today_paths()

######################################
#           Detection
######################################

DETECTION_THRESHOLD = 0.8


def save_license_car_plate(plate_crop, plate_text):
    unique_image_name = f"{uuid1()}.jpg"

    imwrite(os.path.join(IMAGE_FOLDER_PATH, unique_image_name), plate_crop)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=" ", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow([unique_image_name, plate_text])

    return True


def detect_plates(video_path: str, weights: str):
    capture = VideoCapture(video_path)

    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=weights, trust_repo=True
    )

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            print("Video is full processed")
            break

        detections = model(frame)

        for pred_tensor in detections.xyxy:
            if pred_tensor.size(dim=0) == 0:
                continue
            prediction = pred_tensor.tolist()[0]

            confidencee = prediction[4]
            if confidencee < DETECTION_THRESHOLD:
                continue

            xmin = int(prediction[0])
            ymin = int(prediction[1])
            xmax = int(prediction[2])
            ymax = int(prediction[3])

            # 1) !!! Можно не пытаться распознать текст номера, если длина его изображения меньше 250px
            plate_crop = frame[ymin : ymax + 1, xmin : xmax + 1]

            # 2) Обработка, выделение белого и черного: dialate(plate_crop), erode(plate_crop)
            plate_text = recognize_text_with_easyocr(plate_crop)
            if plate_text == "":
                continue

            save_license_car_plate(plate_crop, plate_text)
