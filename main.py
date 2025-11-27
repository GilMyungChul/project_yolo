from ultralytics import YOLO
import cv2

# 모델 로드 (yolov8n-pose, yolov8s-pose, yolov8m-pose 등 선택 가능)
model = YOLO("yolov8s-pose.pt")  # s 버전은 속도/정확도 밸런스 좋음

# 입력 이미지
img_path = "test.jpg"

# GPU 자동 인식 (device='cuda' 강제 가능)
results = model.predict(img_path, device='cuda')

# 결과 그리기
res = results[0]
annotated = res.plot()

cv2.imwrite("pose_result.jpg", annotated)
print("완료! pose_result.jpg 에 저장됨.")