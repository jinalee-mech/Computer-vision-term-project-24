import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from code.vision_utils import extract_tag_info, extract_nutri_info
from code.yolo_utils import calculate_sharpness, crop_box, load_yolo_model
from code.tts_utils import speak_text

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # YOLO 모델 로드
        self.model = load_yolo_model('best2.pt')  # 초기 YOLOv8 모델 경로 (학습 모델 파일)
        self.product_model = load_yolo_model('product.pt')  # 제품모드 YOLOv8 모델 경로 (학습 모델 파일)
        self.current_model = self.model  # 현재 사용 중인 모델

        # 입력 영상 파일
        self.video_path = 'test2.mp4'  # 기본 영상 경로
        self.cap = cv2.VideoCapture(self.video_path)

        # 초기 변수
        self.prev_frame = None
        self.pixel_diffs = []  # 프레임 간 차이값 저장
        self.CALIBRATION_FRAMES = 30  # 초기 캘리브레이션 프레임 수
        self.MOVEMENT_THRESHOLD = 18598152  # 실험적으로 지정
        self.NO_MOVEMENT_DURATION = 2  # 정지 상태 지속 시간 (초)
        self.no_movement_start = None
        self.frame_count = 0  # 처리된 프레임 수를 추적
        self.tag_found = False  # tag 라벨 인식 여부
        self.nutri_found = False  # nutri 라벨 인식 여부
        self.hand_detected = False  # hand 라벨 인식 여부

        # 누적된 객체 저장
        self.detected_objects = {'tag': None, 'nutri': None, 'hand': None}

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle('코너모드')
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

        # 프레임 너비 계산 (왼쪽, 정면, 오른쪽을 나누기 위해)
        frame_width = frame.shape[1]
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        # 그레이스케일 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 코너모드가 ON일 때 움직임이 없을 때만 객체 검출
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(self.prev_frame, gray_frame)
            diff_sum = np.sum(frame_diff)  # 픽셀 차이의 총합
            #print(diff_sum)

            # 움직임 여부 판단
            if diff_sum < self.MOVEMENT_THRESHOLD:
                if self.no_movement_start is None:
                    self.no_movement_start = time.time()
                elif time.time() - self.no_movement_start >= self.NO_MOVEMENT_DURATION:
                    print("Camera movement stopped for 2 seconds. Running object detection...")

                    # YOLO 모델로 객체 감지
                    results = self.current_model(frame, verbose=False)
                    self.detect_objects(frame, results)
                    self.display_frame(frame, self.detected_label)

                    # 정지 상태 트리거 초기화
                    self.no_movement_start = None
            else:
                self.no_movement_start = None

        self.prev_frame = gray_frame

        # 원본 프레임을 QLabel에 표시
        self.display_frame(frame, self.original_label)

        # hand 클래스가 감지되면 제품모드로 전환
        if self.hand_detected:
            print("제품모드")
            self.current_model = self.product_model
            self.frame_count = 0  # 프레임 수 초기화
            self.tag_found = False  # tag 라벨 인식 여부 초기화
            self.nutri_found = False  # nutri 라벨 인식 여부 초기화
            self.hand_detected = False  # hand 라벨 인식 여부 초기화

    def process_product_mode(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        # 원본 프레임을 QLabel에 표시
        self.display_frame(frame, self.original_label)

        # 제품모드가 ON일 때 실시간 객체 검출
        results = self.current_model(frame, verbose=False)
        self.detect_objects(frame, results)

        # 각 라벨이 한 번씩 인식되었고 10프레임을 처리한 경우 OCR과 crop 결과 표시 후 제품모드 OFF로 전환
        if self.frame_count >= 10 and self.tag_found and self.nutri_found:
            best_tag = self.select_best_crop(self.detected_objects, 'tag')
            best_nutri = self.select_best_crop(self.detected_objects, 'nutri')

            if best_tag and best_nutri:
                self.display_crop(best_tag[1], self.tag_crop_label)
                tag_info = extract_tag_info(best_tag[1])
                self.ocr_result.append("제품정보:\n" + tag_info)
                speak_text("제품정보: " + " ".join(tag_info.split()[:5]))

                self.display_crop(best_nutri[1], self.nutri_crop_label)
                nutri_info = extract_nutri_info(best_nutri[1])
                self.ocr_result.append("\n영양 정보:\n" + nutri_info)
                speak_text("영양 정보: " + " ".join(nutri_info.split()[:5]))


                # 제품모드 OFF로 전환
                self.current_model = self.model
                print("코너모드")

    def clear_display(self):
        self.original_label.clear()
        self.detected_label.clear()
        self.tag_crop_label.clear()
        self.nutri_crop_label.clear()
        self.ocr_result.clear()

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

    def detect_objects(self, frame, results):
        # 객체 정보 구분
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])  # 클래스 ID
                confidence = box.conf[0]  # 신뢰도
                label = self.current_model.names[class_id]  # 클래스 이름
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding Box 좌표

                crop = crop_box(frame, (x1, y1, x2, y2))
                if label in self.detected_objects:
                    if self.detected_objects[label] is None or self.detected_objects[label][2] < confidence:
                        self.detected_objects[label] = (label, crop, confidence)

                # 디버깅 출력
                print(f"Detected {label}")

                # 감지된 객체 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} ({confidence:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 + text_size[1] + 10 if y1 + text_size[1] + 10 < frame.shape[0] else y1 - 10
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 라벨 인식 여부 업데이트
                if label == 'tag':
                    self.tag_found = True
                elif label == 'nutri':
                    self.nutri_found = True
                elif label == 'hand':
                    self.hand_detected = True

        self.frame_count += 1

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

    def select_best_crop(self, detected_objects, class_name):
        if detected_objects[class_name] is None:
            return None
        return detected_objects[class_name]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    ex.show()
    sys.exit(app.exec_())