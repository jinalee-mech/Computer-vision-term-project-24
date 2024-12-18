import sys
import cv2
import numpy as np
import time
import os
import pyttsx3
import threading
import queue
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from google.cloud import vision
from google.cloud.vision_v1 import types
from ultralytics import YOLO  # YOLOv8 모델 사용

# Google Vision API 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "charged-sector-444906-k4-51a2ddac9acf.json"

# YOLO 모델 불러오기
model = YOLO('product.pt')  # 학습된 YOLO 모델 경로

# Crop 저장 경로 설정
output_dir = 'output_crops'
os.makedirs(output_dir, exist_ok=True)

# TTS 엔진 초기화
tts_engine = pyttsx3.init()
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()

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

# 이미지 선명도 계산 (라플라시안 방법)
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# 박스 크롭 함수
def crop_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]


def process_detections(frame, detections, alpha=0.6, conf_threshold=0.80, sharpness_threshold=300):
    """
    alpha: 신뢰도와 선명도 가중치 (0.0 <= alpha <= 1.0)
    sharpness_threshold: 선명도의 최소 임계값.
    """
    best_crop = None
    best_score = -1

    for detection in detections:
        box = detection[:4].tolist()
        conf = detection[4].item()

        if conf < conf_threshold:  # 신뢰도 필터링
            continue

        cropped = crop_box(frame, box)

        # 선명도 계산
        sharpness = calculate_sharpness(cropped)
        if sharpness < sharpness_threshold:  # 선명도가 너무 낮으면 무시
            continue

        # 종합 점수 계산
        score = alpha * conf + (1 - alpha) * sharpness

        # 최적의 Crop 업데이트
        if score > best_score:
            best_score = score
            best_crop = cropped

    return best_crop, best_score

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # YOLO 모델 로드
        self.model = YOLO('corner.pt')  # 초기 YOLOv8 모델 경로 (학습 모델 파일)
        self.product_model = YOLO('product.pt')  # 제품모드 YOLOv8 모델 경로 (학습 모델 파일)
        self.current_model = self.model  # 현재 사용 중인 모델

        # 모델을 GPU로 이동
        self.model.to('cuda')
        self.product_model.to('cuda')

        # 입력 영상 파일
        self.video_path = 'demo2.mp4'  # 기본 영상 경로
        self.cap = cv2.VideoCapture(self.video_path)

        # 초기 변수
        self.prev_frame = None
        self.pixel_diffs = []  # 프레임 간 차이값 저장
        self.CALIBRATION_FRAMES = 30  # 초기 캘리브레이션 프레임 수

        self.no_movement_start = None

        # 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 3초 후에 타이머 시작
        self.start_timer = QTimer(self)
        self.start_timer.setSingleShot(True)
        self.start_timer.timeout.connect(self.start_processing)
        self.start_timer.start(5000)  # 3초 대기

        # TTS 타이머 설정
        self.tts_timer = QTimer(self)
        self.tts_timer.setInterval(500)  # 2초 간격
        self.tts_timer.setSingleShot(True)
        self.tts_timer.timeout.connect(self.read_tts_queue)

        # 제품모드 전환 지연 타이머 설정
        self.product_mode_delay_timer = QTimer(self)
        self.product_mode_delay_timer.setInterval(10000)  # 5초 간격
        self.product_mode_delay_timer.setSingleShot(True)
        self.product_mode_delay_timer.timeout.connect(self.enable_product_mode)

        # 크롭 저장 변수
        self.tag_crops = []
        self.nutri_crops = []
        # 라벨 인식 여부 초기화
        self.tag_found = False
        self.nutri_found = False
        self.hand_detected = False
        # 프레임 수 초기화
        self.frame_count = 0
        # 제품모드 전환 가능 여부 초기화
        self.can_switch_to_product_mode = True
        # 코너 읽기 플래그 초기화
        self.corner_read = False

        self.SHARPNESS_THRESHOLD = 0
        self.NO_MOVEMENT_DURATION = 0.5

        self.tag_crops = []
        self.nutri_crops = []
        self.original_tag_crops = []
        self.original_nutri_crops = []

    def start_processing(self):
        self.timer.start(10)  # 30ms마다 프레임 업데이트

    def enable_product_mode(self):
        self.can_switch_to_product_mode = True
        self.hand_detected = False  # 제품모드 전환 가능 시 초기화
        self.corner_read = False  # 코너 읽기 플래그 초기화
        print("코너읽기 false")
        print(self.corner_read)
        print( self.tts_timer.isActive())
        

    def read_tts_queue(self):
        if not tts_queue.empty():
            text = tts_queue.get()
            # TTS 읽기 함수 호출 (예: pyttsx3 또는 다른 TTS 라이브러리 사용)
            print(f"TTS: {text}")  # 디버깅용 출력
            tts_thread = threading.Thread(target=self.tts_speak, args=(text,))
            tts_thread.start()

    def tts_speak(self, text):
        tts_engine.say(text)
        tts_engine.runAndWait()
        self.tts_timer.start()

    def initUI(self):
        self.setWindowTitle('뷰메이트')
        self.setGeometry(100, 100, 1280, 960)  # 창 크기 고정

        self.original_label = QLabel(self)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(640, 480)  # QLabel 크기 고정

        self.detected_label = QLabel(self)
        self.detected_label.setAlignment(Qt.AlignCenter)
        self.detected_label.setFixedSize(640, 480)  # QLabel 크기 고정

        self.tag_crop_label = QLabel(self)
        self.tag_crop_label.setAlignment(Qt.AlignCenter)
        self.tag_crop_label.setFixedSize(320, 240)  # QLabel 크기 고정

        self.nutri_crop_label = QLabel(self)
        self.nutri_crop_label.setAlignment(Qt.AlignCenter)
        self.nutri_crop_label.setFixedSize(320, 240)  # QLabel 크기 고정

        self.ocr_result = QTextEdit(self)
        self.ocr_result.setReadOnly(True)
        self.ocr_result.setFixedSize(320, 240)  # QTextEdit 크기 고정

        layout = QHBoxLayout()
        layout.addWidget(self.original_label)
        layout.addWidget(self.detected_label)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.tag_crop_label)
        right_layout.addWidget(self.nutri_crop_label)
        right_layout.addWidget(self.ocr_result)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def update_frame(self):
        
        if self.current_model == self.product_model:
            self.process_product_mode()
        else:
            self.process_corner_mode()

    def process_corner_mode(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return
        
        self.display_frame(frame, self.original_label)

        frame_width = frame.shape[1]
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is not None:
            sharpness = calculate_sharpness(frame)
            # print(f"Sharpness: {sharpness}")  # 주석 처리

            if sharpness > self.SHARPNESS_THRESHOLD:
                if self.no_movement_start is None:
                    self.no_movement_start = time.time()
                elif time.time() - self.no_movement_start >= self.NO_MOVEMENT_DURATION:
                    print("Camera sharpness is high for 2 seconds. Running object detection...")

                    results = self.current_model(frame, verbose=False)
                    self.detect_objects(frame, results)

                    self.no_movement_start = None
            else:
                self.no_movement_start = None

        self.prev_frame = gray_frame



        if self.hand_detected and self.can_switch_to_product_mode:
            print("제품모드")
            self.current_model = self.product_model
            self.frame_count = 0
            self.tag_found = False
            self.nutri_found = False
            self.hand_detected = False

    def process_product_mode(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        self.display_frame(frame, self.original_label)

        results = self.current_model(frame, verbose=False)
        self.detect_objects(frame, results)

        if self.frame_count > 0:
            self.frame_count += 1

        if self.frame_count >= 120 and self.tag_found and self.nutri_found:
            best_tag = self.select_best_crop(self.original_tag_crops)
            best_nutri = self.select_best_crop(self.original_nutri_crops)

            if best_tag and best_nutri:
                self.display_crop(best_tag[0], self.tag_crop_label)
                tag_info = extract_tag_info(best_tag[0])
                self.ocr_result.append("\n제품정보:\n" + tag_info)
                tts_queue.put("제품정보: " + " ".join(tag_info.split()[:5]))

                self.display_crop(best_nutri[0], self.nutri_crop_label)
                nutri_info = extract_nutri_info(best_nutri[0])
                self.ocr_result.append("\n영양 정보:\n" + nutri_info)
                tts_queue.put("영양 정보: " + " ".join(nutri_info.split()[:5]))

                # 제품모드 OFF로 전환 및 5초 동안 전환 불가 설정
                self.current_model = self.model
                self.can_switch_to_product_mode = False
                self.product_mode_delay_timer.start()
                print("코너모드")

                # OCR을 수행한 후 초기화
                self.tag_crops = []
                self.nutri_crops = []
                self.original_tag_crops = []
                self.original_nutri_crops = []

    def display_frame(self, frame, label):
        # 프레임을 QImage로 변환
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # QLabel 크기에 맞게 영상 크기 조정
        scaled_qt_image = qt_image.scaled(label.size(), Qt.KeepAspectRatio)

        # QLabel에 이미지 설정
        label.setPixmap(QPixmap.fromImage(scaled_qt_image))

    def detect_objects(self, frame, results, max_crops=10, alpha=0.5):
       

        for result in results:
            boxes = result.boxes.data
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = self.current_model.names[class_id]

                if conf < 0.5:
                    continue

                # Crop 원본 이미지 저장
                original_crop = frame[int(y1):int(y2), int(x1):int(x2)]  # 라벨링 전 원본 crop
             
                


                if label == 'tag':
                    self.tag_found = True
                    processed, score = process_detections(frame, [box], alpha)
                    if processed is not None:
                        self.tag_crops.append((processed, score))
                        self.original_tag_crops.append((original_crop.copy(), score))
                    if self.frame_count == 0:
                        self.frame_count = 1  # 첫 detect 후 frame_count 시작
                elif label == 'nutri':
                    self.nutri_found = True
                    processed, score = process_detections(frame, [box], alpha)
                    if processed is not None:
                        self.nutri_crops.append((processed, score))
                        self.original_nutri_crops.append((original_crop.copy(), score))
                    if self.frame_count == 0:
                        self.frame_count = 1  # 첫 detect 후 frame_count 시작
                elif label == 'hand':
                    self.hand_detected = True
                elif label in ['snack', 'beverage', 'instant', 'ramen']:
                    if not self.corner_read :
                        tts_queue.put(f"{label} corner")
                        self.read_tts_queue()
                        self.corner_read = True  # 코너 읽기 플래그 설정
                        print("코너읽기 true")

                print(f"Detected {label}")

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{label} ({conf:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = int(x1)
                text_y = int(y1) + text_size[1] + 20 if int(y1) + text_size[1] + 20 < frame.shape[0] else int(y1) - 10
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # 신뢰도가 0.5 이상인 프레임을 오른쪽 위에 표시
                self.display_frame(frame, self.detected_label)



    def display_crop(self, crop, label):
        # 크롭된 이미지를 QImage로 변환
        if isinstance(crop, np.ndarray):
            rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QLabel 크기에 맞게 영상 크기 조정
            scaled_qt_image = qt_image.scaled(label.size(), Qt.KeepAspectRatio)

            # QLabel에 이미지 설정
            label.setPixmap(QPixmap.fromImage(scaled_qt_image))

    def select_best_crop(self, crops):
        best_crop = None
        
        for crop, score in crops:
            if best_crop is None or score > best_crop[1]:
                best_crop = (crop, score)
        
        return best_crop

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    ex.show()
    sys.exit(app.exec_())