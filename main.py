import os
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

import csv
from uuid import uuid1

from cv2 import VideoCapture, imread, imwrite, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
import torch
from easyocr import Reader

from modules.detection.detect import detect_plate
from modules.recognition.recognize import recognize_text_with_easyocr

######################################
#             ROOT path
######################################

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

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
#      main()
######################################

VID_FORMATS = "mp4", "mkv", "mpg", "mpeg", "mov", "gif"
IMG_FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")


def save_license_car_plate(plate_crop, plate_text):
    unique_image_name = f"{uuid1()}.jpg"

    imwrite(os.path.join(IMAGE_FOLDER_PATH, unique_image_name), plate_crop)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=" ", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow([unique_image_name, plate_text])


def video_pipeline(source, imgsz, model, reader):
    time = datetime.now()
    print(f"\n\t2. Detection and recognition started (image size {imgsz})\n")

    capture = VideoCapture(source)

    fps = capture.get(CAP_PROP_FPS)
    frames = int(capture.get(CAP_PROP_FRAME_COUNT))
    duration = frames / fps

    frame_count = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            print("Video is full processed")
            break

        time_for_one_frame = datetime.now()
        frame_count += 1

        plate_crop = detect_plate(frame, imgsz, model)
        if plate_crop is None:
            print(
                f"Frame №{frame_count}, {datetime.now() - time_for_one_frame}   no detections"
            )
            continue

        plate_text = recognize_text_with_easyocr(plate_crop, reader)
        if plate_text == "" or len(plate_text) < 6:
            print(
                f"Frame №{frame_count}, {datetime.now() - time_for_one_frame}   the number wasn't recognized, or car plate is too small"
            )
            continue

        save_license_car_plate(plate_crop, plate_text)

        print(
            f'Frame №{frame_count}, {datetime.now() - time_for_one_frame}   "{plate_text}"'
        )
    print(
        f"\n\tDetection and recognition finished!\n\tVideo duration: {duration} seconds\n\tElapsed {datetime.now() - time} for car plate detection, recognition and saving"
    )


def image_pipeline(source, imgsz, model, reader):
    image = imread(source)

    plate_crop = detect_plate(image, imgsz, model)
    if plate_crop is None:
        return ""
    else:
        return recognize_text_with_easyocr(plate_crop, reader)


def test_pipeline(imgsz, model, reader):
    import json

    with open(f"{ROOT}/test/dataset.json", "r", encoding="utf-8") as labels_file:
        test_labels = json.load(labels_file)["labels"]

    all_images = len(test_labels)

    right_preditctions = 0
    for label in test_labels:
        label_text = label["nums"][0]["text"]
        label_file = f'{ROOT}/test/images/{label["file"]}'

        pred_text = image_pipeline(label_file, imgsz, model, reader)

        print(f"{label_text}, {pred_text.upper()}")

        if label_text == pred_text.upper():
            right_preditctions += 1

    precision = right_preditctions / all_images
    print(f"Точность: {precision}")

    return precision


def main(
    source=f"{ROOT}/data/sample.mp4",
    weights=f"{ROOT}/models/y5m_baseline.pt",
    imgsz=640,
):
    is_video = Path(source).suffix[1:] in VID_FORMATS
    is_image = Path(source).suffix[1:] in IMG_FORMATS
    is_test = Path(source).exists and (source == "test")

    if not is_video and not is_image and not is_test:
        print(
            f"[--source N] Expected VIDEO, IMAGE file or 'test' ('test' folder should exist), but got {source}"
        )
        return

    print("\n\t1. Downloading models ...\n")

    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=weights, trust_repo=True
    )

    reader = Reader(["ru"])

    print("\n\tDownload - success!")

    if is_video and Path(source).is_file():
        video_pipeline(source, imgsz, model, reader)

    if is_image and Path(source).is_file():
        number = image_pipeline(source, imgsz, model, reader)
        print(f'\n\t{source} -> license car number: "{number}"')

    if is_test:
        # python main.py --weights .\models\y5s6.pt --source test --img 1280
        test_pipeline(imgsz, model, reader)

    # is_url =
    # TODO()
    print("\n\tExit ...")
    return "zxc"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=f"{ROOT}/models/y5m_baseline.pt",
        help="Trained model *.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=f"{ROOT}/data/sample.mp4",
        help="Source file (video) or link (stream)",
    )
    parser.add_argument(
        "--imgsz",
        "--imgs",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="inference size h,w",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
