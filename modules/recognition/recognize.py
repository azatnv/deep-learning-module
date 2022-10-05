from dataclasses import asdict
import numpy as np
import cv2 as cv

######################################
#   Recognition of license number
######################################

RECOGNITION_THRESHOLD = 0.7

MINIMUM_PIXEL_PLATE_LENGHT = 50  # ПОРОГ 250px !!!!!!!!!!!!!!!!!!
# Поставил маленький порог, чтобы для записей sample3-4.mp4 была хоть какая-то детекция.
# для sample2.mp4, где Женя впритык снимает номера, можно поставить 150-250px
# В идеале нам нужно видео в котором, будут крупные номера, как в sample2.mp4


# Example of list_with_all_recognitions = [([[12, 12], [38, 12], [38, 42], [12, 42]], 'Р', 0.9999908209057686), ([[35, 0], [224, 0], [224, 45], [35, 45]], '056ХР56', 0.9997990848715271)]
def filter_text(list_with_all_recognitions) -> str:
    # return list_with_all_recognitions[0][1].upper().replace(" ", "")
    interesting_elements = [
        (x[0], x[1].upper().replace(" ", ""), x[2])
        for x in list_with_all_recognitions
        if x[2] > RECOGNITION_THRESHOLD
    ]

    match len(interesting_elements):
        case 0:
            return ""
        case 1:
            return interesting_elements[0][1]
        case 2:
            first = interesting_elements[0][1]
            second = interesting_elements[1][1]
            if first.isnumeric():
                return "".join([second, first])
            else:
                return "".join([first, second])
        case _:
            max_word_index = -1
            max_word_length = -1

            for i, recogniton_result in enumerate(interesting_elements):
                if max_word_length < len(recogniton_result[1]):
                    max_word_index = i
                    max_word_length = len(recogniton_result[1])

            maxword_and_secondword = [max_word_index, -1]
            second_word_probability = -1

            for i, recogniton_result in enumerate(interesting_elements):
                if (
                    i != maxword_and_secondword[0]
                    and second_word_probability < recogniton_result[2]
                ):
                    maxword_and_secondword[1] = i
                    second_word_probability = recogniton_result[2]

            if (
                maxword_and_secondword[0] > 0
                and maxword_and_secondword[1] > 0
                and len(interesting_elements[0][1]) == 1
            ):
                return "".join(
                    [
                        interesting_elements[0][1],
                        interesting_elements[maxword_and_secondword[0]][1],
                        interesting_elements[maxword_and_secondword[1]][1],
                    ]
                )

            match len(interesting_elements[maxword_and_secondword[1]][1]):
                case 1:
                    return "".join(
                        [
                            interesting_elements[maxword_and_secondword[1]][1],
                            interesting_elements[maxword_and_secondword[0]][1],
                        ]
                    )
                case _:
                    return "".join(
                        [
                            interesting_elements[maxword_and_secondword[0]][1],
                            interesting_elements[maxword_and_secondword[1]][1],
                        ]
                    )


def dialate(image):
    element = cv.getStructuringElement(cv.MORPH_RECT, (2, 2), (-1, -1))  # 2,2 или 3,3
    # element = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4), (-1, -1))  # 3,3 или 4,4
    # element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3), (-1, -1))  # 3,3

    return cv.dilate(image, element)


def erode(image):
    element = cv.getStructuringElement(cv.MORPH_RECT, (2, 2), (-1, -1))

    return cv.erode(image, element)


def recognize_text_with_easyocr(image: np.ndarray, reader) -> str:
    if image.shape[1] < MINIMUM_PIXEL_PLATE_LENGHT:
        return ""

    # print(image.shape)

    # Предобработка
    image = dialate(image)
    image = erode(image)

    results = reader.readtext(
        image,
        allowlist="0123456789АВЕКМНОРСТУХ",
        text_threshold=0.5,
        contrast_ths=0.9,
        adjust_contrast=1,
    )

    if len(results) == 0:
        return ""

    return filter_text(results)
