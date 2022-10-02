from cv2 import VideoCapture, imwrite
import os
import csv
from uuid import uuid1
import torch

from modules.recognition.recognize import recognize_text_with_easyocr

from main import CSV_PATH, IMAGE_FOLDER_PATH, DETECTION_THRESHOLD


model = torch.hub.load("ultralytics/yolov5", "custom", path="./models/yolov5x.pt")


def save_license_car_plate(plate_crop, plate_text):
    unique_image_name = f"{uuid1()}.jpg"

    imwrite(os.path.join(IMAGE_FOLDER_PATH, unique_image_name), plate_crop)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow([unique_image_name, plate_text])

    return True


def detect_plates(video_path: str):
    capture = VideoCapture(video_path)

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

            plate_crop = frame[ymin : ymax + 1, xmin : xmax + 1]
            plate_text = recognize_text_with_easyocr(plate_crop)
            if plate_text == "":
                continue

            save_license_car_plate(plate_crop, plate_text)
