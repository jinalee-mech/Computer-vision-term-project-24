import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8 모델 사용

# YOLO 모델 불러오기
def load_yolo_model(model_path):
    return YOLO(model_path)

# 이미지 선명도 계산 (라플라시안 방법)
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# 박스 크롭 함수
def crop_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]