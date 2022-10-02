import os
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

import torch

from modules.detection.detect import detect_plates

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
#            CONSTANTS
######################################

VID_FORMATS = "mp4", "mkv", "mpg", "mpeg", "gif"

DETECTION_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 0.5
# REGION_THRESHOLD = 0.4

######################################
#      main()
######################################


def main(weights=f"{ROOT}/models/yolov5x.pt", source=f"{ROOT}/data/sample.mp4"):

    is_file = Path(source).suffix[1:] in VID_FORMATS

    if is_file and Path(source).is_file():
        time = datetime.now()
        detect_plates(source)
        print(
            f"Elapsed {datetime.now() - time} for car plate detection, recognition and saving"
        )
    else:
        print(f"Expected VIDEO file, but got {source}")
        return

    # print(weights, source)
    # print(Path(weights).exists(), Path(source).exists())

    print(
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        torch.cuda.get_device_name(0),
    )

    # is_url =
    # TODO()
    return "zxc"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=f"{ROOT}/models/yolov5x.pt",
        help="Trained model *.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=f"{ROOT}/data/sample.jpg",
        help="Source file (video) or link (stream)",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
