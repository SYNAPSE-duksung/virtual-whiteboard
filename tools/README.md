# 데이터 수집 도구 안내 

`tools/` 폴더의 스크립트로 학습용 데이터를 수집하고 채점하는 방법을 정리한 문서입니다. 노션에도 이 문서를 그대로 공유합니다.

## 1. 개요

- **`record_dual.py`**: 메인(사선) + 측면(스마트폰) 카메라를 동시에 모니터링하며 녹화하는 PyQt GUI 도구입니다.
- **`extract_landmarks.py`**: `record_dual`로 찍은 main 영상에서 오프라인으로 랜드마크·pen_ratio·pen_down을 뽑아 학습용 CSV를 만듭니다.
- **`label_frames.py`**: 측면 영상을 보며 pen down/up 정답 라벨을 토글 방식으로 매기는 GUI 도구입니다.
- **`estimate_camera_offset.py`** (7주차): 메인·측면 영상의 손 움직임을 상호상관해 `label_frames.py`의 카메라 지연 오프셋을 자동 추정합니다.
- **`calibrate.py`** (A, 4주차): 카메라로 책상 위 필기 영역의 네 모서리를 클릭해 `core.geometry.PerspectiveCalibration`을 계산·저장하는 독립 실행 도구입니다. 같은 4점 지정 로직(`core.geometry.CalibrationPicker`)이 `controller/main.py`(OpenCV 데모, `K` 키)에도 내장되어 있어 세션 안에서 바로 (재)캘리브레이션할 수 있습니다.
- **`validate_normalization.py`**, **`validate_z_axis.py`** (A, 3주차): 합성 3D 손 모델로 정규화 수식·Z축 임계값을 실촬영 없이 검증하는 스크립트입니다. D 데이터가 쌓이면 같은 프레임워크(`RatioSpec`)에 실데이터를 물려 ROC까지 확장할 수 있습니다.

## 2. record_dual 사용법

배경: 카메라 2대로 동시에 찍습니다. **메인 = 노트북 웹캠**(사선 각도, 학습 입력 feature용), **측면 = 스마트폰**(USB 연결, 책상 높이 옆모습에서 손끝-책상 접촉이 보이도록 배치 — 정답 라벨 판정용)입니다.

1. 실행: `python -m tools.record_dual`
2. 폼에 촬영자 이름, 조명(`bright`/`dark`), 속도(`slow`/`fast`), 메인/측면 카메라 인덱스를 입력합니다.
3. 어떤 인덱스가 어느 카메라인지 모르면 **"카메라 검색"** 버튼으로 사용 가능한 인덱스와 해상도를 확인합니다.
4. **"카메라 열기"**를 눌러 두 카메라 프리뷰를 띄웁니다. (측면 카메라가 없어도 메인만으로 녹화 가능합니다 — 이 경우 상태바에 "측면 없음" 안내가 뜹니다.)
5. **"검출 테스트"**를 켜서 메인 프리뷰에 손 검출 결과(초록 랜드마크 + HAND OK/NO HAND)가 뜨는지 확인하세요. **NO HAND가 자주 뜨면 녹화해도 랜드마크 추출이 안 되니**, 웹캠 각도·거리·조명을 조정해 HAND OK가 안정적으로 뜨는 구도를 잡은 뒤 녹화하세요. (녹화를 시작하면 자동으로 꺼지고, 영상에는 표시가 남지 않습니다.)
6. **"● 녹화 시작"** → 촬영 → **"■ 녹화 정지"**.

저장물은 `data/dataset/recordings/` 아래에 3종이 생깁니다 (파일명 규칙: `이름_조명_속도_회차`):

- `이름_조명_속도_회차_main.mp4` — 사선 뷰 (이후 랜드마크 추출용)
- `이름_조명_속도_회차_side.mp4` — 측면 뷰 (접촉 여부 정답 라벨용)
- `이름_조명_속도_회차_frames.csv` — 프레임별 타임스탬프 + main/side 수신 여부 (두 영상 싱크·프레임 누락 검출용)

같은 조건(이름/조명/속도)으로 다시 찍으면 회차 번호가 자동으로 올라갑니다.

## 3. 스마트폰을 서브카메라로 연결하기 (플랫폼별) 
** 노션에 자세히 적어놓겠습니다.

- **Windows + Android (웹캠 모드가 없는 기기)**: **Iriun Webcam** 사용 — 폰(Play 스토어)과 PC(iriun.com) 양쪽에 설치하고 둘 다 실행. 

## 4. 촬영 프로토콜

1. **싱크 마커**: 녹화 시작 직후 카메라 두 대에 다 보이게 박수를 1회 칩니다. (나중에 두 영상을 프레임 단위로 맞추는 기준점입니다. 라벨링 때 츠레임 싱크 맞추는 작업 진행 예정)
2. **측면 폰 배치**: 책상 높이에서 손 옆모습이 보이도록, 손끝과 책상 접촉면이 화면에 나오도록 놓습니다.
3. **조건 매트릭스**: 촬영자(팀원 교대) × 조명(`bright`/`dark`) × 속도(`slow`/`fast`) 조합당 최소 1회씩 찍습니다.
4. **영상 파일은 git에 올리지 않습니다.** `data/dataset/recordings/`는 `.gitignore`에 등록되어 있으며, 구글 드라이브로 공유합니다.

## 5. extract_landmarks 사용법

`record_dual`은 녹화 중 프레임 드랍을 막기 위해 랜드마크를 추출하지 않습니다. 이 스크립트가 저장된 main 영상을 다시 읽어 오프라인으로 좌표를 계산합니다. 

```
python -m tools.extract_landmarks
```

`data/dataset/recordings/`의 모든 `*_main.mp4`를 훑어 영상마다 CSV 2종을 만듭니다 (둘 다 이미 있으면 건너뜁니다):

- `{base}_coords.csv` — 손끝 좌표·pen_ratio·pen_down (아래 스키마)
- `{base}_landmarks.csv` — **손 랜드마크 21개 전체**(mediapipe 정규화 x,y,z). 스키마: `frame_id, timestamp, hand_detected, x0, y0, z0, …, x20, y20, z20, video_id`. 각도변화 같은 새 feature를 만들 때 사용하고, 손 미검출 프레임은 좌표가 빈 값입니다.

- 개별 파일/폴더 지정, `--overwrite`(기존 결과 덮어쓰기)도 가능합니다.
- 옵션: `--min-cutoff`, `--beta` (One Euro Filter 파라미터), `--pen-down-thresh`, `--pen-up-thresh` (pen 판정 히스테리시스 임계값), `--min-detection-confidence`/`--min-tracking-confidence` (MediaPipe 신뢰도 — 손 검출률이 낮게 나오면 `--min-detection-confidence 0.5 --overwrite`로 재추출해 보세요).
- `--calibration output/calibration.json` — 4점 캘리브레이션을 지정하면 `rectified_x`/`rectified_y`(필기 평면 기준 정류 좌표)와 `in_bounds`(평면 안쪽 여부는 pen_down 컬럼에는 반영되지 않고 좌표만 채워짐) 컬럼까지 함께 계산합니다. 지정하지 않으면 두 컬럼은 기존처럼 빈 값입니다. **듀얼 카메라 글자 복원 계획**(`CLAUDE.md` 참고)의 `tools/reconstruct_stroke.py`가 이 컬럼을 입력으로 씁니다.
- CSV 스키마는 `core/recorder.py`의 `CSV_COLUMNS`에 `video_id`(마지막 컬럼, 값=base명, 예: `minjin_bright_slow_01`)가 추가된 형태입니다: `frame_id, timestamp, hand_detected, raw_x, raw_y, raw_z, filtered_x, filtered_y, pen_ratio, pen_down, rectified_x, rectified_y, video_id`. `video_id`는 여러 영상의 CSV를 합칠 때 영상을 구분하고, 시계열 feature(속도/가속도/저크) 계산 시 프레임 순서를 보존하며, 클립 단위로 train/test를 분리하는 데 씁니다.
- `frame_id`는 main 영상의 프레임 번호이며, 같은 회차의 `_frames.csv`(타임스탬프 + main/side 수신 여부)를 매개로 측면 영상 프레임과 정렬됩니다. 측면 영상으로 라벨링한 결과를 이 기준으로 매칭하면 됩니다.

## 6. label_frames 사용법

측면(`*_side.mp4`) 영상을 보면서 pen down/up 정답 라벨을 만드는 GUI 도구입니다. mediapipe 없이 영상과 CSV만으로 동작합니다.

```
python -m tools.label_frames
```

- **화면 구성**: 우측 큰 화면 = 측면 영상(판정 기준, 현재 상태 DOWN/UP 오버레이), 좌측 위 = 같은 시점 메인 카메라 화면(문맥 확인), 좌측 아래 = 토글 지점 목록(삭제·±1프레임 수정), 하단 = 슬라이더 + down 구간이 빨갛게 칠해진 타임라인.
- **방식**: 토글 지점 마킹 — 시작은 무조건 UP이며, 상태가 바뀌는 순간 `T`를 눌러 전환점을 찍으면 그 지점부터 상태가 뒤집힙니다. 프레임별 클릭은 필요 없습니다.
- **키**: `SPACE` 재생/정지, 재생 속도 0.5×/1×/2×, `←`/`→` 1프레임, `Shift+←`/`Shift+→` 10프레임, `T` 토글, `Ctrl+Z` 마지막 토글 취소, `Ctrl+S` 저장.
- **출력**: `{base}_labels.csv` (`video_id, frame_id, pen_down`, 프레임당 1행). `{base}_coords.csv`와 `video_id + frame_id`로 조인하면 학습 데이터가 완성됩니다. 기존 라벨 파일이 있으면 불러와 이어서 편집합니다.
- **참고**: 프레임 누락이 있어도 `_frames.csv`를 매개로 전역 frame_index에 정렬되므로 coords와 어긋나지 않습니다.

## 7. estimate_camera_offset 사용법 (카메라 지연 자동 추정, 7주차)

`label_frames.py`의 "지연 오프셋(프레임)" 스핀박스를 눈으로 맞추는 대신, 메인·측면 각
영상의 손 움직임을 상호상관(cross-correlation)해 자동으로 추정합니다. 박수 같은 동기화
마커가 없어도 되고(촬영 내내 일어나는 자연스러운 손 움직임 자체가 동기화 신호), 첫
toggle을 마커로 쓰려는 시도가 구조적으로 안 되는 이유(측면에서 본 toggle에 대응하는
"메인에서 본 같은 순간"을 눈으로 찾을 수 없음)를 아예 피해갑니다.

```
python -m tools.estimate_camera_offset 이름_조명_속도_회차 --save
```

- 인자 없이 실행하면 `data/dataset/recordings/` 아래 모든 `*_main.mp4`를 일괄 처리합니다.
- **원리**: 메인·측면 각 영상에 독립적인 `HandTracker`를 돌려(같은 인스턴스를 두 스트림에
  번갈아 먹이면 내부 트래킹 상태가 섞이므로 반드시 분리) 검지 끝의 정규화 좌표 기준
  프레임간 변위(움직임 크기) 시계열을 뽑고, 균일 시간축으로 리샘플한 뒤 상호상관으로
  가장 잘 겹치는 시차를 찾습니다.
- **출력값은 `label_frames.py`의 오프셋과 정의가 정확히 같습니다**
  (`g = side_map[k] - offset`). `--save`를 주면 `labeling_offsets.json`에 바로 기록되어
  `label_frames.py`를 열 때 자동 복원됩니다.
- **신뢰도 확인**: 결과에 정규화 상관계수(-1~1)가 함께 출력됩니다. **0.3 미만이면 낮은
  신뢰도로 경고**가 뜨니 그런 영상은 `label_frames.py`에서 육안으로 검산하세요. 손이 오래
  멈춰 있거나 움직임이 단조로운 영상에서 특히 낮게 나올 수 있습니다.
- **`--plot`**: 정렬 전/후 신호와 lag별 상관계수 그래프(`{base}_offset_debug.png`)를 함께
  저장합니다. 추정이 못 미더울 때 왜 그런 값이 나왔는지 눈으로 확인할 수 있습니다.
- **검색 범위**: 기본 `--max-lag-sec 2.0`(±2초). 실제 지연이 이보다 크면 못 잡으니, 결과의
  `offset_frames_unclamped`가 스핀박스 범위(±30프레임)를 벗어난다는 경고가 뜨면
  `--max-lag-sec`를 늘려 재시도하세요.
- 같은 촬영 세션·같은 물리적 연결(폰 위치·케이블/Iriun 설정 안 바꿈)이면 오프셋을 세션
  전체 영상에 재사용해도 된다는 게 이 프로젝트의 기존 가정입니다 — 처음 몇 개만
  `label_frames.py`에서 교차 확인한 뒤, 나머지는 이 도구로 일괄 처리해도 됩니다.

## 8. reconstruct_stroke 사용법 (듀얼 카메라 글자 복원, 7주차)

메인(노트북) 카메라의 궤적(정류 좌표)과 측면(폰) 카메라의 자동 pen up/down 신호를
타임스탬프로 결합해 손글씨를 복원하는 도구입니다. 자세한 배경·설계는 `CLAUDE.md`의
"듀얼 카메라 기반 pen up/down 개선 + 글자 복원 실험 계획" 참고.

**사전 준비**:
1. `record_dual`로 `{base}_main.mp4`/`{base}_side.mp4`/`{base}_frames.csv`를 녹화
2. `python -m tools.extract_landmarks --calibration output/calibration.json`으로
   `{base}_coords.csv`에 정류 좌표(`rectified_x`/`rectified_y`)까지 포함해 추출
   (`--calibration` 없이 추출한 기존 CSV는 이 컬럼이 비어 있어 그대로 쓸 수 없습니다 —
   `--overwrite`로 재추출하세요)

```
python -m tools.reconstruct_stroke
```

- `data/dataset/recordings/`의 모든 `*_side.mp4`를 훑어 `{base}_reconstructed.png`
  (흰 배경, 검은 잉크)를 만듭니다.
- **측면 기준선**: 책상면을 나타내는 직선을 처음 한 번 지정해야 합니다. 저장된
  `output/side_baseline.json`이 없으면 첫 영상의 첫 프레임에서 마우스로 책상 가장자리
  위 2점을 클릭 → `Enter`/`s`로 확정(`z` 취소, `Esc` 중단). `--save-baseline`으로
  저장해두면 같은 카메라 구도에서 다음 실행 때 재사용됩니다. 카메라를 옮겼다면
  `--recalibrate-baseline`으로 다시 지정하세요.
- **헤드리스 지정**: `--baseline X1 Y1 X2 Y2`로 클릭 없이 좌표를 직접 줄 수도 있습니다.
- **접촉 문턱**: `--down-px`/`--up-px`(기본 20/35px)는 카메라-책상 거리·해상도에 따라
  달라지는 임시값입니다. 실측 후 `core.touch_calibration.estimate_thresholds()`로
  hover/touch 표본 기반 재추정을 적용하는 것이 다음 단계입니다.
- **품질 확인**: `--ocr`을 주면 복원 결과를 EasyOCR로 인식해 텍스트를 출력합니다
  (`easyocr` 설치 필요, 기본은 꺼져 있어 무거운 모델 로드를 하지 않습니다). 원문과
  비교해 복원이 되는지(CLAUDE.md 계획의 1단계 판단 기준)를 정량적으로 확인하세요.
- **출력 요약**: 실행 후 콘솔에 메인 CSV 행 수, 측면 프레임 수·검출률·pen-down 비율,
  잘라낸 stroke 개수가 출력됩니다.

## 9. calibrate 사용법 (A, 4주차)

책상 위 필기 영역의 네 모서리(사선 각도로 찍힌 사다리꼴)를 클릭으로 지정해, 위에서 내려다본 것처럼 반듯하게 펴는 투시 변환(`cv2.getPerspectiveTransform`/`warpPerspective`)을 계산·저장하는 도구입니다.

```
python -m tools.calibrate
```

- **순서**: 화면을 좌클릭해 **TL(좌상단) → TR(우상단) → BR(우하단) → BL(좌하단)** 순서로 4점을 찍습니다. 순서가 꼬여 자기교차(나비꼴) 사각형이 되거나 점끼리 너무 가까우면 저장 시 오류 메시지가 뜨고 저장되지 않습니다.
- **키**: `z` 마지막 점 취소, `r` 전체 리셋, `p` 정류(bird's-eye) 미리보기 창 토글(4점 완료 후), `s` 저장, `q` 종료.
- **출력**: `output/calibration.json` (경로는 `--output`으로 변경 가능). 정류 캔버스 크기(`dst_size`)는 지정하지 않으면 원본 사각형의 변 길이로 자동 추정됩니다(비율 보존).
- **재사용**: `core.geometry.PerspectiveCalibration.load("output/calibration.json")`으로 다른 스크립트에서 그대로 불러와 `warp_frame()`(이미지 정류), `to_rectified()`/`from_rectified()`(좌표 매핑), `contains()`(점이 필기 영역 안인지) 등을 쓸 수 있습니다.
- **판정 파이프라인 연결됨**: `controller.main.WhiteboardSession`이 세션 시작 시 `output/calibration.json`을 자동 로드하고, `core.PenTracker`가 `contains()`로 손끝이 평면 밖일 때 pen_ratio 판정과 무관하게 강제 pen up으로 게이팅한다(⚠️ 2D 평면 범위 판정이며 진짜 3D 접촉/높이 판정은 아님 — `core/pen_tracker.py` docstring 참고). `extract_landmarks.py`(D의 오프라인 CSV 추출)는 `--calibration`을 지정하면 정류 좌표까지 함께 뽑도록 연결됐다(8절 `reconstruct_stroke` 참고).
- **재캘리브레이션**: `controller/main.py` 데모는 캘리브레이션이 없으면(최초 실행) 시작 시 자동으로 지정을 강제하고, 이후에는 `K` 키로 언제든 다시 캘리브레이션할 수 있습니다(적용 전까지 기존 캘리브레이션은 유지). 카메라 위치·화각이 바뀌면 재캘리브레이션이 필요합니다 — 촬영 세션마다 언제 다시 찍을지는 10절 협의 대상입니다.

## 10. [제안 — D와 페어세션에서 확정] 캘리브레이션 데이터 협의 초안

> 이 절은 A가 실제로 D와 만나 확정한 내용이 아니라, **페어세션에서 논의할 안건을 미리 정리한 초안**입니다. 세션 후 합의된 내용으로 이 절을 덮어써 주세요.

**`calibration.json` 스키마** (이미 구현·고정됨, `core/geometry.py`):

```json
{
  "version": 1,
  "src_points": [[x,y], [x,y], [x,y], [x,y]],  // TL,TR,BR,BL, 원본 카메라 픽셀 좌표
  "dst_size": [width, height],                  // 정류 캔버스 크기(px)
  "created_at": 1234567890.0,
  "source_frame_size": [width, height]           // 캘리브레이션 당시 카메라 프레임 크기(참고용)
}
```

**논의할 것**:

1. **D의 CSV에 정류 좌표를 추가할 가치가 있는가?** `extract_landmarks.py`가 캘리브레이션 파일을 읽어 `rectified_x`, `rectified_y`(손끝을 `to_rectified()`로 변환한 값) 컬럼을 `{base}_coords.csv`에 추가할 수 있습니다. 사선 왜곡이 제거된 좌표라 시계열 feature(속도 등)가 더 안정적일 수 있는데, ML 모델 입력으로 실제 도움이 되는지는 실험이 필요합니다.
2. **촬영 세션당 캘리브레이션을 언제 찍는가?** 카메라가 고정이면 세션 시작 시 1회로 충분하지만, `record_dual.py` 촬영마다 카메라 위치가 달라지면 매 회차 재캘리브레이션이 필요합니다 — 촬영 프로토콜(4절)에 캘리브레이션 단계를 넣을지 결정이 필요합니다.
3. **`contains()`(필기 영역 내부 판정)를 라벨링에 쓸 수 있는가?** `label_frames.py`가 측면 영상으로 pen down/up을 라벨링할 때, `contains()`로 "손끝이 애초에 필기 영역 밖"인 프레임을 걸러내면 라벨링 품질에 도움이 될 수 있습니다.
4. **파일 공유 위치**: `calibration.json`은 `output/`(gitignore)에 저장되므로 git으로 공유되지 않습니다 — 팀이 공유할 필요가 있다면 구글 드라이브 등 별도 채널이 필요합니다(3절의 영상 파일과 동일한 문제).
