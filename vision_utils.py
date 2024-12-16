import cv2
import numpy as np
import os
from google.cloud import vision
from google.cloud.vision_v1 import types

# Google Vision API 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "operating-bird-444206-q9-7a6fa808b86e.json"

# Google Vision OCR 함수
def google_vision_ocr(image):
    client = vision.ImageAnnotatorClient()

    # 이미지를 메모리에서 읽어서 Google Vision API에 전달
    if isinstance(image, np.ndarray):
        success, encoded_image = cv2.imencode('.jpg', image)
        content = encoded_image.tobytes()
        image = types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description  # 전체 텍스트 반환
    return ""

# OCR 및 키워드 추출 함수 (제품정보)
def extract_tag_info(cropped_image):
    text = google_vision_ocr(cropped_image)
    words = text.split()  # 단어로 나누기
    product_info = " ".join(words[:20])  # 앞에서 20단어 선택
    return product_info

def extract_nutri_info(cropped_image):
    return google_vision_ocr(cropped_image)  # 영양 정보 전체 반환