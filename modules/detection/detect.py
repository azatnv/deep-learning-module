from modules.recognition.recognize import recognize_plate
from cv2 import VidoeCapture


def detect_and_recognize(path: str):
    # 1. Read video with VidoeCapture(path)

    # 2. detect car number in one frame

    # 3. recognize text on plate
    recognize_plate()

    return "TODO()"
