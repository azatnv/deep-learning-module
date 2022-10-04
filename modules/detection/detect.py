######################################
#       Detection of car plate
######################################

CONFIDENCE_THRESHOLD = 0.5  # default 0.25 for /yolo/detect.py (0.001 for /yolo/val.py)
IOU_THRESHOLD = 0.45  # default 0.45 for /yolo/detect.py (0.6 for /yolo/val.py)


def detect_plate(img, img_size, model):
    model.conf = CONFIDENCE_THRESHOLD  # NMS confidence threshold
    model.iou = IOU_THRESHOLD  # NMS IoU threshold
    model.max_det = 1  # maximum number of detections per image

    detection = model(img, size=img_size)

    pred_tensor = detection.xyxy[0]
    if pred_tensor.size(dim=0) == 0:
        return None

    prediction = pred_tensor.tolist()[0]

    xmin = int(prediction[0])
    ymin = int(prediction[1])
    xmax = int(prediction[2])
    ymax = int(prediction[3])

    return img[ymin : ymax + 1, xmin : xmax + 1]
