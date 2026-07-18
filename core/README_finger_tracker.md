# SYNAPSE MVP — 손가락 추적 + 드로잉 + CSV 추출

## 설치

```bash
pip install -r requirements.txt
```

(mediapipe는 numpy 버전에 민감합니다. 설치 오류 시 `pip install "numpy<2"` 후 재시도)

## 실행
프로젝트 루트에서
```bash
python -m core.finger_tracker
```

옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--camera` | 0 | 웹캠 인덱스 |
| `--min-cutoff` | 1.0 | One Euro Filter, 낮을수록 더 부드럽지만 지연 증가 |
| `--beta` | 0.3 | 빠른 움직임 반응성, 높을수록 빠른 움직임에서 필터 강도 약화 |
| `--pen-down-thresh` | 0.55 | 이 값보다 작으면 펜다운(그리기 시작) |
| `--pen-up-thresh` | 0.70 | 이 값보다 크면 펜업(그리기 중단) — 히스테리시스로 떨림 방지 |
| `--output-dir` | output | CSV 저장 폴더 |
| `--no-record` | off | 시작 시 자동 기록을 끔 (`r` 키로 수동 시작) |

## 조작법

- `q` : 종료 (CSV 저장 후 닫힘)
- `r` : CSV 기록 시작/일시정지 토글 (새로 시작할 때마다 새 CSV 파일 생성)
- `c` : 캔버스 지우기
- 손을 쫙 펴면(4손가락 모두 폄) 자동으로 캔버스가 지워짐

## 구현 요약 (제안서 대응)

1. **검지(8번) 실시간 좌표 추출**: MediaPipe Hands로 프레임마다 landmark 8(Index Finger Tip)의 정규화 좌표를 픽셀 좌표로 변환.
2. **필터 적용**: One Euro Filter(저지연 지터 제거 필터)를 x, y 각각에 독립 적용. 정지 시엔 부드럽게, 빠르게 움직일 때는 민첩하게 반응.
3. **Pen-up/down 판정**: 제안서의 "① 랜드마크 간 거리 비율" 방식 채택 — Tip(8)-DIP(7) y좌표 차이를 손 크기(wrist-middle_mcp 거리)로 정규화. 히스테리시스(진입 임계값 < 복귀 임계값)로 경계에서의 떨림 방지.
4. **CSV 추출**: 매 프레임마다 `frame_id, timestamp, hand_detected, raw_x/y/z, filtered_x/y, pen_ratio, pen_down`을 기록. 30프레임마다 flush하여 중간에 종료돼도 데이터 유실 최소화.

## 다음 단계 제안 (제안서 로드맵 기준)

- z축 임계값(보조 신호)을 pen_ratio와 결합해 pen-down 정확도 개선 여부 테스트
- 4점 투시 변환(getPerspectiveTransform/warpPerspective) 캘리브레이션 단계 추가 → 사선 각도 보정
- OpenCV `adaptiveThreshold`로 필기 궤적 후처리 → EasyOCR 입력 전처리
- 수집된 CSV로 pen_ratio 임계값을 데이터 기반으로 재조정 (여러 사용자/거리에서 샘플 수집 후 분포 확인)

## 참고

- 카메라/디스플레이가 없는 환경(서버, 컨테이너 등)에서는 `cv2.imshow`가 동작하지 않습니다. 로컬 PC(웹캠 있는 환경)에서 실행하세요.
