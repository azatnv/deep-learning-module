import os
import sys
import argparse
from datetime import datetime
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
#      main()
######################################

VID_FORMATS = "mp4", "mkv", "mpg", "mpeg", "gif"


def main(source=f"{ROOT}/data/sample.mp4", weights=f"{ROOT}/models/y5m_baseline.pt"):

    is_file = Path(source).suffix[1:] in VID_FORMATS

    if is_file and Path(source).is_file():
        time = datetime.now()
        detect_plates(source, weights)
        print(
            f"Elapsed {datetime.now() - time} for car plate detection, recognition and saving"
        )
    else:
        print(f"Expected VIDEO file, but got {source}")
        return

    print(
        torch.cuda.is_available(),
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
        default=f"{ROOT}/models/y5m_baseline.pt",
        help="Trained model *.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=f"{ROOT}/data/sample.mp4",
        help="Source file (video) or link (stream)",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
