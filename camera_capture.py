import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_image():
    # 카메라 객체 생성
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return None
    
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 카메라 해제
    cap.release()
    
    if ret:
        return frame
    return None

def display_image(image):
    if image is not None:
        # BGR에서 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # matplotlib으로 이미지 표시
        plt.figure(figsize=(10, 6))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()
    else:
        print("이미지를 표시할 수 없습니다.")

def main():
    print("카메라로 이미지를 캡처합니다...")
    image = capture_image()
    display_image(image)

if __name__ == "__main__":
    main() 