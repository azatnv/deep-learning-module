import os
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

import csv
from uuid import uuid1

from cv2 import VideoCapture, imwrite, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
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

VID_FORMATS = "mp4", "mkv", "mpg", "mpeg", "gif"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"


def save_license_car_plate(plate_crop, plate_text):
    unique_image_name = f"{uuid1()}.jpg"

    imwrite(os.path.join(IMAGE_FOLDER_PATH, unique_image_name), plate_crop)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=" ", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow([unique_image_name, plate_text])


def main(
    source=f"{ROOT}/data/sample.mp4",
    weights=f"{ROOT}/models/y5m_baseline.pt",
    imgsz=640,
):
    is_video = Path(source).suffix[1:] in VID_FORMATS
    is_image = Path(source).suffix[1:] in IMG_FORMATS

    if not is_video and not is_image:
        print(f"Expected VIDEO or IMAGE file, but got {source}")
        return

    print("\n\t1. Downloading models ...\n")

    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=weights, trust_repo=True
    )

    reader = Reader(["ru"])

    print("\n\tDownload - success!")

    if is_video and Path(source).is_file():
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
                continue

            save_license_car_plate(plate_crop, plate_text)

            print(
                f'Frame №{frame_count}, {datetime.now() - time_for_one_frame}   "{plate_text}" shape={plate_crop.shape[0:2]}'
            )
        print(
            f"""\n\tDetection and recognition finished!
        Video duration: {duration} seconds
        Elapsed {datetime.now() - time} for car plate detection, recognition and saving
        """
        )

    # is_url =
    # TODO()
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
