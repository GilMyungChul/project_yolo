import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# 거리 계산 함수
# -----------------------------
def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


# -----------------------------
# 체형 분류 함수 (남/여 기준)
# -----------------------------
def classify_body_shape(shoulder_width, hip_width, torso_length, leg_length, gender):
    shoulder_hip_ratio = shoulder_width / hip_width
    leg_torso_ratio = leg_length / torso_length

    # -----------------------------
    # 남성 기준
    # -----------------------------
    if gender == "male":
        if shoulder_hip_ratio >= 1.25:
            return "역삼각형(상체 넓음)"
        if shoulder_hip_ratio <= 0.90:
            return "삼각형(하체 중심)"
        if 0.90 < shoulder_hip_ratio < 1.25:
            if 0.8 <= leg_torso_ratio <= 1.1:
                return "직사각형(기본 남성 체형)"
            if leg_torso_ratio > 1.1:
                return "모래시계형"
        return "판단 불가"

    # -----------------------------
    # 여성 기준
    # -----------------------------
    if gender == "female":
        if shoulder_hip_ratio >= 1.15:
            return "역삼각형(상체 발달)"
        if shoulder_hip_ratio <= 0.85:
            return "삼각형(하체 발달)"
        if 0.85 < shoulder_hip_ratio < 1.15:
            if 0.9 <= leg_torso_ratio <= 1.2:
                return "직사각형(허리선 약함)"
            if leg_torso_ratio > 1.2:
                return "모래시계"
        return "판단 불가"

    return "성별 오류"


# -----------------------------
# 시각화: 어깨선/골반선/체형 텍스트
# -----------------------------
def visualize(image, kpts, shape_result):
    img = image.copy()

    # 어깨선 (5,6)
    ls, rs = kpts[5], kpts[6]
    cv2.line(img, tuple(ls.astype(int)), tuple(rs.astype(int)), (0, 255, 0), 3)

    # 골반선 (11,12)
    lh, rh = kpts[11], kpts[12]
    cv2.line(img, tuple(lh.astype(int)), tuple(rh.astype(int)), (255, 0, 0), 3)

    # 결과 텍스트
    cv2.putText(img, f"Body Shape: {shape_result}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return img


# -----------------------------
# 메인 파이프라인
# -----------------------------
def analyze_body(image_path, gender="female"):
    # YOLO pose 모델 로드
    model = YOLO("yolov8s-pose.pt")

    # 이미지 읽기
    image = cv2.imread(image_path)
    results = model.predict(image, device='cuda')[0]

    if results.keypoints is None:
        print("사람을 찾지 못했습니다.")
        return

    kpts = results.keypoints.xy[0].cpu().numpy()  # (17,2)

    # 필요한 keypoints
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_KNEE, RIGHT_KNEE = 13, 14
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16

    # 길이 계산
    shoulder_width = dist(kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER])
    hip_width = dist(kpts[LEFT_HIP], kpts[RIGHT_HIP])

    leg_left = dist(kpts[LEFT_HIP], kpts[LEFT_KNEE]) + dist(kpts[LEFT_KNEE], kpts[LEFT_ANKLE])
    leg_right = dist(kpts[RIGHT_HIP], kpts[RIGHT_KNEE]) + dist(kpts[RIGHT_KNEE], kpts[RIGHT_ANKLE])
    leg_length = (leg_left + leg_right) / 2

    torso_length = dist((kpts[LEFT_SHOULDER] + kpts[RIGHT_SHOULDER]) / 2,
                        (kpts[LEFT_HIP] + kpts[RIGHT_HIP]) / 2)

    # 체형 분석
    result = classify_body_shape(shoulder_width, hip_width, torso_length, leg_length, gender)

    print("-------- 체형 분석 결과 ---------")
    print("어깨 너비:", shoulder_width)
    print("골반 너비:", hip_width)
    print("상체 길이:", torso_length)
    print("다리 길이:", leg_length)
    print("결과:", result)
    print("--------------------------------")

    # 시각화
    vis = visualize(image, kpts, result)
    cv2.imwrite("body_shape_result.jpg", vis)
    print("body_shape_result.jpg 저장 완료!")


# -----------------------------
# 실행 예시
# -----------------------------
if __name__ == "__main__":
    IMAGE_PATH = "test.jpg"

    # analyze_body(IMAGE_PATH, gender="female")    # 여성이면 female
    analyze_body(IMAGE_PATH, gender="male")    # 남성이면 male
