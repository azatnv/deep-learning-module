import os
import argparse
import sys
from pathlib import Path
from modules.detection.detect import detect_and_recognize

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

VID_FORMATS = "mp4", "mkv", "mpg", "mpeg", "gif"


def main(weights=f"{ROOT}/models/yolov5x.pt", source=f"{ROOT}/data/sample.jpg"):
    is_file = Path(source).suffix[1:] in VID_FORMATS
    if is_file and Path(source).is_file():
        detect_and_recognize(source)

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
