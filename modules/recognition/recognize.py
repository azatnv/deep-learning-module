from easyocr import Reader
import numpy as np

reader = Reader(["ru"])

RECOGNITION_THRESHOLD = 0.5


def filter_text(recognition_list) -> str:
    # Переписать эту функцию, тк она выводит текст только с наибольшего региона.
    # наибольший регион - это сам номер Р123ОТ
    # не хватает региона
    max_region_size = 0
    max_index = -1
    for i, result in enumerate(recognition_list):
        if result[2] < RECOGNITION_THRESHOLD:
            continue
        width = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        region_size = width * height

        if region_size > max_region_size:
            max_region_size = region_size
            max_index = i

    if max_index == -1:
        return ""

    return recognition_list[max_index][1]


def recognize_text_with_easyocr(image: np.ndarray) -> str:
    results = reader.readtext(image)

    # print(f"{results}\n")

    if len(results) == 0:
        return ""

    return filter_text(results)
