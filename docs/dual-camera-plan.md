# 듀얼 카메라 기반 pen up/down 개선 + 글자 복원 실험 계획

> 7주차~. 관련 이슈: [#32](https://github.com/SYNAPSE-duksung/virtual-whiteboard/issues/32).
> 구현 위치는 `CLAUDE.md`의 디렉토리 구조·아키텍처 규칙 절에도 반영돼 있다.

## 배경

5주차까지의 3D 접촉 판정(`core/contact3d.py`)은 **노트북 카메라 하나**(사선/거의 수직으로
책상을 내려다보는 구도)로 pen up/down과 글자 궤적을 동시에 뽑으려 한 시도였다. 실측 결과
재투영 오차가 자주 튀었고, 원인 분석 결과 이 구도 자체가 두 목적에 서로 다른 요구를
건다는 게 드러났다:

- **pen up/down 판정**: 손끝-책상 간격을 봐야 하는데, 위에서 내려다보면 그 간격이
  거의 안 보이고(원근 축 방향과 겹침), 손을 책상 쪽으로 숙이는 자세는 하필 자기가림
  (self-occlusion)이 심해 MediaPipe world landmarks 노이즈가 가장 큰 자세와 겹친다.
- **글자 궤적 복원**: 반대로 이 구도는 필기 평면을 정면으로 보므로 4점 캘리브레이션
  → `warpPerspective` 정류가 잘 맞고, XY 궤적 복원에는 유리하다.

즉 **하나의 카메라로 두 목적을 동시에 잘 하기 어렵다는 것이 원인**이라는 가설을 세우고,
카메라 2대로 역할을 나눠 각각 잘하는 일만 시키는 방향으로 전환한다.

이미 D파트 학습 데이터 수집용으로 구축된 **메인(노트북)+측면(폰) 동시 녹화 인프라**
(`tools/record_dual.py`, `tools/label_frames.py`, 촬영 프로토콜은 `tools/README.md`
4절)가 이 역할 분담과 정확히 같은 카메라 배치다 — 다만 지금까지는 측면 영상을
**사람이 수동으로 라벨링**(`label_frames.py`)해 ML 학습용 정답을 만드는 데만 썼다.
이번 계획은 그 측면 영상에서 pen up/down을 **자동으로** 뽑아 실제 글자 복원 파이프라인에
쓰는 것으로 목적을 확장한다.

## 카메라 역할 분담

| | 메인(노트북, 사선/탑다운) | 측면(폰, 책상 높이 옆모습) |
|---|---|---|
| 담당 | 글자 궤적(트래킹) | pen up/down 판정 |
| 이유 | 필기 평면이 정면으로 보여 4점 캘리브레이션·정류가 정확 | 손끝-책상 접촉 간격이 원근 왜곡 없이 직접 보임, 손가락 가림도 덜함 |
| 재사용 가능 모듈 | `core/tracker.py`, `core/geometry.py`(`to_rectified`) | 신규 구현 — `core/side_contact.py` |
| 어려운 점 | pen up/down (5주차까지 확인된 문제) | 글자 모양 복원(평면 정면이 아니라 XY 왜곡, 시야각 제한) |

## 산출물 정의

한 회차 녹화(`record_dual.py`)당 아래 3종 데이터를 CSV로 남긴다.

1. **원본 글자 좌표 CSV** — 메인 영상에서 뽑은 손끝 픽셀 좌표. `tools/extract_landmarks.py`가
   `raw_x, raw_y, filtered_x, filtered_y`를 `{base}_coords.csv`로 뽑는다.
2. **인식된 평면 정보** — 4점 캘리브레이션 결과. `core/geometry.py`의
   `PerspectiveCalibration`(`src_points`, `dst_size`, `plane_size_mm`)을 그 회차 촬영
   구도로 한 번 잡아 `output/calibration.json`으로 저장 (`tools/calibrate.py`).
3. **평면 연산대로 변환한 데이터(정류 좌표)** — 원본 좌표를 (2)의 캘리브레이션으로
   `to_rectified()`한 결과. `tools/extract_landmarks.py --calibration`으로 채운다.

## 진행 순서

### 1단계 — 메인 궤적 + 측면 pen 신호만으로 글자 복원이 되는지 우선 판단

- [x] **정류 좌표 추출** — `tools/extract_landmarks.py --calibration`으로 `calibration.json`을
      로드해 원본 좌표마다 정류 좌표(`rectified_x, rectified_y`) 컬럼을 함께 뽑도록 확장
      (`core.geometry.PerspectiveCalibration.to_rectified()` 재사용). `core/recorder.py`의
      `CSV_COLUMNS`/`CoordSample`에도 같은 컬럼을 추가해 라이브 세션(`controller/main.py`)의
      CSV 기록에도 자동으로 반영됨.
- [x] **측면 영상에서 pen up/down을 자동으로 뽑는 신규 모듈** `core/side_contact.py`.
      노트북 카메라의 `contact3d.py`(이중 solvePnP)보다 훨씬 단순한 문제라는 판단대로,
      **책상면 기준선 하나만 캘리브레이션**(2점, `SideBaselinePicker`)해 그 선까지의
      픽셀 거리(`SideBaseline.distance_of`)에 히스테리시스(`SideContactDetector`)를
      씌우는 방식으로 구현. 접촉 문턱(`--down-px`/`--up-px`)은 아직 실측 기반이 아닌
      자리표시자(20/35px)이며, `core.touch_calibration.estimate_thresholds()` 재적용은
      후속 작업.
- [x] **두 스트림 결합** — `tools/reconstruct_stroke.py`. `_frames.csv`의 `side_ok` 플래그를
      순서대로 세어 측면 영상 프레임과 elapsed_sec을 정확히 매핑(단순히 `frame_index`를
      쓰면 메인/측면 드랍이 다를 때 어긋남)하고, 메인 CSV 각 행에 가장 가까운 측면 표본을
      최근접 타임스탬프로 매칭해 정류 좌표를 stroke로 잘라 `ui.canvas.StrokeCanvas`에
      렌더링, `{base}_reconstructed.png`로 저장. 측면 기준선은 대화형 2점 클릭 또는
      `--baseline X1 Y1 X2 Y2`(헤드리스)로 지정하고 `--save-baseline`으로 재사용 가능.
- [x] **복원 품질 판단 기준** — `--ocr` 플래그로 복원 결과를 EasyOCR에 통과시켜 인식
      텍스트를 출력(선택 기능, 지연 import로 무거운 모델 로드를 기본 경로에서 피함).
      **아직 안 된 것**: 실제 카메라로 녹화 → 복원 → 원문과 비교하는 실측 검증 — 지금까지는
      코드 경로를 합성 데이터로만 검증함.
- [x] **실시간 연동** — `core/pen_tracker.py`가 `side_frame`을 받아 측면 신호를
      **측면(최우선) → 3D → pen_ratio** 순으로 적용하도록 배선됨. `controller/main.py`
      (`--side-camera N`, `B` 키로 대화형 기준선 지정 + SIDE 창)와 `ui/app.py`
      (`--side-camera N`, 상태 표시만 — 대화형 지정은 controller에서만 가능)에 연결됨.
- [x] **카메라 간 시간 정렬 자동화** — `tools/estimate_camera_offset.py`. 메인·측면 각
      영상의 손 움직임(검지 끝 프레임간 변위)을 독립적으로 추출해 상호상관으로 카메라
      지연을 자동 추정, `label_frames.py`의 "지연 오프셋(프레임)" 스핀박스 값을
      `labeling_offsets.json`에 미리 채워둔다. 박수 같은 동기화 마커 없이도 동작하며,
      정규화 상관계수가 낮게 나오면(0.3 미만) 경고해 육안 검산을 유도한다.

### 2단계 — 1단계로 복원이 어려울 경우: 두 카메라 데이터를 함께 쓰는 모델로 최종 판단

- [ ] 메인 궤적 feature(위치·속도·곡률 등) + 측면 feature(손끝-책상 간격, 손 각도 등)를
      프레임 단위로 결합한 학습 데이터셋 구성 — `label_frames.py`로 이미 만들어 둔
      수동 라벨을 정답으로 재사용 가능
      - **⚠️ 정책 확인 필요**: 프로젝트의 "딥러닝 프레임워크 금지" 규칙은 D파트의
        pen up/down "상태 판정 모델"에 한정된 제약이다. 2단계의 "최종 글자 판단"이
        프레임 단위 상태 판정의 연장(그러면 scikit-learn 유지)인지, 시퀀스/문자 인식에
        더 가까운 별도 과업(그러면 딥러닝 허용 여부를 팀/지도교수와 재확인)인지 착수
        전에 확정할 것.

## 남은 리스크 / 미해결 질문

- **실카메라 미검증**: 폰+노트북 두 대를 실제로 동시에 구동했을 때 프레임레이트·지연시간
  차이가 실사용에 문제가 되는지 검증되지 않았다 — 카메라가 없는 환경이라 지금까지는
  합성 프레임으로 로직만 확인했다(코드 경로 검증 ≠ 실기기 검증). 문제가 크면
  `record_dual.py` 녹화 → `reconstruct_stroke.py` 오프라인 검증으로 되돌아가는 것도
  선택지로 남겨둔다.
- 측면 카메라의 캘리브레이션(책상면 기준선)이 촬영 때마다 손으로 다시 잡아야 하는지,
  고정 거치대를 전제로 한 번만 하면 되는지는 촬영 프로토콜과 함께 확정 필요.
- 이 계획은 기존 5주차 3D 접촉 판정(`core/contact3d.py`, 노트북 단일 카메라)을
  당장 대체하는 게 아니라 **병행 실험**이다 — 1단계 결과가 나쁘면 기존 단일 카메라
  파이프라인도 계속 유지·개선한다.
- **접촉 신호 실측 검증 아직 없음**: `core/side_contact.py`·`tools/reconstruct_stroke.py`는
  코드 경로(직선 거리 계산·히스테리시스·타임스탬프 매칭·stroke 렌더링)를 합성 데이터로만
  검증했다 — `tools/validate_contact3d.py`가 3D 접촉 신호에 대해 한 것과 같은 실측
  검증(노이즈·분리도 d′)이 이 신호에는 아직 없다. 실제 듀얼 카메라 녹화본으로
  (1) 접촉 문턱(`--down-px`/`--up-px`) 기본값이 맞는지 (2) 복원 stroke가 실제로
  읽히는지(`--ocr`)를 확인하는 게 다음으로 할 일이다.
- **오프셋 자동 추정의 전제**: 손의 움직임 패턴이 두 카메라에서 상관돼 있다는 가정에
  의존한다. 손이 오래 정지해 있거나 움직임이 단조로운 영상에서는 상관계수가 낮게 나올 수
  있다 — 그런 영상은 여전히 `label_frames.py`에서 육안 검산이 필요하다.

## 관련 파일

| 파일 | 역할 |
|---|---|
| `core/side_contact.py` | 측면 카메라 기준선·접촉 판정 |
| `core/pen_tracker.py` | 측면 신호를 3D/pen_ratio보다 우선 적용하는 실시간 판정 로직 |
| `controller/main.py`, `ui/app.py` | 실시간 듀얼 카메라 연동 (`--side-camera`, `B` 키) |
| `core/recorder.py`, `tools/extract_landmarks.py` | 정류 좌표 CSV 컬럼 |
| `tools/reconstruct_stroke.py` | 오프라인 stroke 복원 + OCR 품질 확인 |
| `tools/estimate_camera_offset.py` | 카메라 간 시간 정렬 자동 추정 |
