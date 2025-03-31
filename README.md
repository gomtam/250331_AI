# 실시간 객체 인식 프로그램

이 프로젝트는 웹캠을 통해 실시간으로 객체를 인식하고 분류하는 프로그램입니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 프로그램 실행:
```bash
python app.py
```

2. 프로그램 종료:
- ESC 키를 누르면 프로그램이 종료됩니다.

## 주요 기능

- 실시간 웹캠 영상 표시
- 객체 인식 및 분류
- 신뢰도 점수 표시
- 336x336 크기의 확대된 화면 표시

## 주의사항

- 웹캠이 연결되어 있어야 합니다.
- Python 3.7 이상 버전이 필요합니다.
- `model` 폴더에 학습된 모델(`keras_model.h5`)과 레이블 파일(`labels.txt`)이 있어야 합니다.


<img src ="https://github.com/gomtam/image/blob/main/250331/KakaoTalk_20250331_154450850.png" width="800">
