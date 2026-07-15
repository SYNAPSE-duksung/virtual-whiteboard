# 데이터 수집 도구 안내 

`tools/` 폴더의 스크립트로 학습용 데이터를 수집하고 채점하는 방법을 정리한 문서입니다. 노션에도 이 문서를 그대로 공유합니다.

## 1. 개요

- **`record_dual.py`**: 메인(사선) + 측면(스마트폰) 카메라를 동시에 모니터링하며 녹화하는 PyQt GUI 도구입니다.
- **`extract_landmarks.py`**: `record_dual`로 찍은 main 영상에서 오프라인으로 랜드마크·pen_ratio·pen_down을 뽑아 학습용 CSV를 만듭니다.
- **`label_frames.py`**: 측면 영상을 보며 pen down/up 정답 라벨을 토글 방식으로 매기는 GUI 도구입니다.

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
- CSV 스키마는 `core/recorder.py`의 `CSV_COLUMNS`에 `video_id`(마지막 컬럼, 값=base명, 예: `minjin_bright_slow_01`)가 추가된 형태입니다: `frame_id, timestamp, hand_detected, raw_x, raw_y, raw_z, filtered_x, filtered_y, pen_ratio, pen_down, video_id`. `video_id`는 여러 영상의 CSV를 합칠 때 영상을 구분하고, 시계열 feature(속도/가속도/저크) 계산 시 프레임 순서를 보존하며, 클립 단위로 train/test를 분리하는 데 씁니다.
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
