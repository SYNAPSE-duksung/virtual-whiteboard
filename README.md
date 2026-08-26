# Virtual Whiteboard

책상 위에 손가락으로 쓴 글씨를 웹캠으로 인식하는 가상 칠판입니다.
손끝 움직임을 실시간으로 추적해 획을 만들고, OCR과 로컬 LLM으로 글자를 읽어 문맥에 맞게 정리합니다.

<img width="578" height="324" alt="image" src="https://github.com/user-attachments/assets/71c852bb-29ea-41d1-8cf5-cf621df9ad23" />


## 차별점

기존의 필기 인식 방식에는 Air-writing이 있다. 해당 방법은 공중에서 손가락을 움직여 글씨를 쓴다. 이 방식에는 두 가지 문제가 있다.

- 손을 지지할 면이 없어 글씨가 흔들리고, 이미 쓴 위치를 파악하기 어렵다.
- “펜을 뗀다”에 해당하는 물리적 사건이 없어 글자 사이의 획이 모두 이어진다.

이 프로젝트는 책상면을 필기면으로 활용해 두 문제를 동시에 해결한다. 손끝이 실제 책상에 닿기 때문에
필기감이 생기고, “닿음/뗌”이라는 명확한 물리적 사건으로 획의 시작과 끝을 구분할 수 있다.


### 핵심 과제

카메라만으로 손끝이 지금 책상에 닿아 있는지 판단해야 한다.
손끝의 XY 좌표는 픽셀 기준으로 비교적 정확히 얻을 수 있지만, Z축(책상에서 떨어진 높이) 정보는 카메라가 위에서 내려다보는 구도에서는 원근 방향과 겹쳐 거의 드러나지 않는다. 아래의 판정 구조는 이 문제를 해결하기 위한 것이다.

## Pen up/down 판정

`core/pen_tracker.py`는 아래 방법을 우선순위대로 시도하며, 사용할 수 없으면 다음 방법으로 넘어간다.
사용된 신호는 언제든 `PenFrame.contact_source`에서 확인할 수 있다.

| 순위 | 방식 | 신호 | 조건 |
|---|---|---|---|
| **1** | **측면 카메라** | 손끝과 책상 기준선 사이의 거리(px) | `--side-camera` 옵션 + `B` 키로 기준선 2점 지정 |
| **2** | **3D 접촉 판정** | 이중 `solvePnP`로 복원한 **실제 높이(mm)** | 4점 캘리브레이션 + 평면 크기(mm) + intrinsics |
| **3** | **pen_ratio** | 검지 Tip–DIP 세로 거리 ÷ 손 크기 | 항상 사용 가능(최종 폴백) |

2번 방식은 MediaPipe의 `world_landmarks`가 제공하는 미터 단위 3D 좌표를 활용한다. 평면 자세와 손 자세를 각각 `solvePnP`로 추정한 뒤 손끝을 평면 좌표계로 변환하면, z 성분이 책상에서의 높이(mm)가 된다. 계통 오차는 `T` 키의 영점 보정으로 줄인다.

`D` 키(PyQt에서는 “3D 디버그” 버튼)를 누르면 평면 수선 벡터, 높이 눈금자, 임계선, 수치 패널을 영상 위에 겹쳐 볼 수 있다.

## 전체 파이프라인

<img width="1247" height="501" alt="image" src="https://github.com/user-attachments/assets/96f1eb2d-af1c-4033-bbca-3ea881b64334" />



| 단계 | 모듈 |
|---|---|
| 랜드마크 추적 | `core/tracker.py` |
| 좌표 안정화 | `core/filters.py` (OneEuroFilter) |
| 원근 정류 | `core/geometry.py` (4점 캘리브레이션 → `warpPerspective`) |
| Pen 판정 | `core/pen_tracker.py`, `contact3d.py`, `side_contact.py`, `pen_state.py` |
| 상태 안정화 | `controller/state_machine.py` |
| 렌더링 | `ui/canvas.py`, `ui/app.py` |
| OCR·LLM | `controller/ocr_llm_pipeline.py`, `ai/` |

## 환경 구성

Python 3.11 x64를 권장한다. 이 프로젝트는 `mediapipe==0.10.21`을 사용한다.
0.10.35부터 legacy `mp.solutions` API가 제거되므로 이 버전을 유지해야 한다.

```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1        # PowerShell
source venv/Scripts/Activate       # bash

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

```bash
# OpenCV 창 버전
python -m controller.main

# PyQt 앱 버전
python -m ui.app

# 측면 카메라를 함께 사용하는 듀얼 카메라 모드
python -m controller.main --side-camera 1
```

### 키 조작

| 키 | 동작 |
|---|---|
| `M` | 자동/수동 모드 전환 |
| `SPACE` | 수동 모드에서 펜 올리기·내리기 |
| `K` | 4점 캘리브레이션 — `TL→TR→BR→BL` 순서로 클릭(또는 손끝 + `SPACE`) |
| `T` | 접촉 캘리브레이션 — hover/touch 표본으로 임계값과 영점 추정 |
| `B` | 측면 카메라 기준선 지정(책상 가장자리 위 2점) |
| `D` | 3D 디버그 오버레이 토글 |
| `O` | OCR/LLM 파이프라인 실행 |
| `C` / `S` / `Q` | 지우기 / 캔버스 PNG 저장(`captures/`) / 종료 |

자동 모드에서는 손을 편 상태로 0.25초간 유지하면 캔버스가 지워진다.
짧은 오판이나 트래킹 손실이 발생해도 상태머신이 획을 자연스럽게 이어 준다.

### 카메라 구도 옵션

기본값은 **카메라가 책상을 내려다보는 구도**를 전제로 하며, 화면을 반전하지 않는다.

| 옵션 | 용도 |
|---|---|
| `--mirror` | 셀피 구도에서 테스트할 때 |
| `--flip-vertical` | 사선 웹캠 화면이 상하로 뒤집혀 보일 때(`--mirror`와 독립) |
| `--camera N` | 카메라 장치 선택 |
| `--plane-size W H` | 캘리브레이션할 사각형의 실제 크기(mm). 기본값 `297 210`은 가로 방향 A4 |
| `--no-3d` | 3D 판정을 끄고 pen_ratio만 사용 |

## 프로젝트 구조

```
core/          CV·기하학 로직(UI와 독립)
  tracker.py         MediaPipe 랜드마크 추출
  filters.py         OneEuroFilter
  geometry.py        4점 캘리브레이션 + 원근 정류
  camera.py          카메라 intrinsics
  distances.py       손가락 거리 + 손 크기 정규화
  pen_state.py       pen_ratio 휴리스틱
  contact3d.py       3D 접촉 판정(이중 solvePnP)
  side_contact.py    측면 카메라 기준선 판정
  pen_tracker.py     3단계 폴백 파사드
  finger_tracker.py  독립 실행 좌표 수집 MVP

controller/    제어 루프·상태 관리
  session.py         프레임 단위 진입점(OpenCV/PyQt 공용)
  overlay.py         오버레이 렌더링
  state_machine.py   디바운싱·트래킹 손실 브리징
  ocr_llm_pipeline.py 비동기 OCR/LLM 연결부

ui/            canvas.py(렌더링 엔진) · app.py(PyQt)
ai/            EasyOCR 파라미터 튜닝 · LLM 문맥 보정
tools/         데이터 수집·검증 스크립트(아래 참조)
```

### 도구 스크립트

| 스크립트 | 용도 |
|---|---|
| `tools/calibrate.py` | 4점 캘리브레이션 지정 및 정류 결과 미리보기 |
| `tools/calibrate_camera.py` | 체스보드 기반 `camera_intrinsics.json` 생성 |
| `tools/record_dual.py` | 메인·측면 카메라 동시 녹화 GUI |
| `tools/label_frames.py` | 측면 영상을 이용한 pen down/up 라벨링 GUI |
| `tools/extract_landmarks.py` | 녹화 영상에서 랜드마크·좌표 CSV 추출 |
| `tools/estimate_camera_offset.py` | 상호상관으로 두 카메라의 시간차 자동 추정 |
| `tools/reconstruct_stroke.py` | 메인 궤적과 측면 신호 결합 → stroke 복원(+ OCR) |
| `tools/validate_contact3d.py` | 3D 접촉 신호 실측 검증(노이즈 σ, 분리도 d′) |
| `tools/validate_normalization.py` | 정규화 수식 검증(합성 손 모델, 실촬영 불필요) |
| `tools/validate_z_axis.py` | z축 임계값 타당성 실험 |

자세한 사용법은 [`tools/README.md`](tools/README.md)를, 모드와 키 조작은 [`controller/README.md`](controller/README.md)를 참고한다.

## 주요 실험 결과

| 실험 | 결과 |
|---|---|
| 정규화 수식 20종 비교(합성) | 현행 `pen_ratio` 수식은 19위. 손 회전에 취약 |
| z축 임계값 타당성 | 기각. 손가락 마디는 강체이므로 3D 거리가 굽힘에 따라 변하지 않음 |
| 3D 접촉 판정 수학 검증(합성) | 높이 복원 오차 0.00mm |
| 카메라 시간차 자동 추정 | 0.5초 인위 지연 주입 후 +0.500초 복원, 상관계수 1.00 |
| 휴리스틱 vs. ML 판정 비교 | 기존 임계값 58.7% / Logistic Regression 66.4% / RBF SVM 67.1% |

정확도가 67%에서 더 높아지지 않은 원인은 분류기보다 입력 신호에 있다.
사용한 특징이 모두 한 카메라에서 얻은 같은 랜드마크에 기반하기 때문이다.

## 향후 방향

두 카메라가 **같은 종이를 공통 기준물로** 관찰하도록 구성하면, 두 시점의 시선을 교차시켜
스케일 추정 없이 손끝의 실제 높이를 얻을 수 있다(삼각측량). 예상 정확도는 1~2mm로,
접촉 판정 기준인 12mm에 비해 충분한 여유가 있다.

- 흰 종이 윤곽 자동 검출 → 매번 4점을 클릭해야 하는 번거로움 제거
- 손끝 트래킹 강건화 — 검출(MediaPipe) + 추적(광학 흐름) + 예측



## 팀

4인이 역할을 나누어 진행했다. `core/`를 UI와 분리하고 모듈의 독립성을 유지해 병합 충돌을 줄였다.

| 파트 | 담당 |
|---|---|
| CV 코어·기하학 | 랜드마크 추적·필터, 캘리브레이션·정류, pen up/down 판정, 검증 도구 |
| 통합·UI | 메인 제어 루프, StrokeCanvas, PyQt 앱, 상태머신, 비동기 파이프라인 |
| AI 연동 | EasyOCR 파라미터 튜닝, 로컬 LLM 문맥 보정 |
| 데이터·ML | 듀얼 녹화·라벨링 GUI, Feature Engineering, scikit-learn 판정 모델 비교 |
