import numpy as np

######################################
#   Recognition of license number
######################################

RECOGNITION_THRESHOLD = 0.5

MINIMUM_PIXEL_PLATE_LENGHT = 50  # px
# Поставил маленький порог, чтобы для записей sample3-4.mp4 была хоть какая-то детекция.
# для sample2.mp4, где Женя впритык снимает номера, можно поставить 150-250px
# В идеале нам нужно видео в котром, будут крупные номера, как в sample2.mp4


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


def dialate(image):  # В разработке
    return image


def erode(image):  # В разработке
    return image


def recognize_text_with_easyocr(image: np.ndarray, reader) -> str:
    if image.shape[1] < MINIMUM_PIXEL_PLATE_LENGHT:
        return ""

    # Обработка
    # image = dialate(image)
    # image = erode(image)
    results = reader.readtext(image)

    if len(results) == 0:
        return ""

    return filter_text(results)
