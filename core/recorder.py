"""좌표 CSV 기록 (D파트 ML 학습 데이터 수집).

``core.finger_tracker``와 ``controller`` 양쪽에서 동일한 컬럼·포맷으로 CSV를 남기기 위해
기록 로직을 한곳으로 모은 모듈. CV/판정 로직과 독립적이며, 한 프레임의 값을
``CoordSample``로 받아 ``output/coords_<timestamp>.csv``에 기록한다.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CSV_COLUMNS = [
    "frame_id",
    "timestamp",
    "hand_detected",
    "raw_x",
    "raw_y",
    "raw_z",
    "filtered_x",
    "filtered_y",
    "pen_ratio",
    "pen_down",
]

_FLUSH_EVERY = 30  # N프레임마다 디스크 flush


@dataclass
class CoordSample:
    """CSV 한 행에 해당하는 프레임 값."""

    hand_detected: bool
    raw_x: float | None = None
    raw_y: float | None = None
    raw_z: float | None = None
    filtered_x: float | None = None
    filtered_y: float | None = None
    pen_ratio: float | None = None
    pen_down: bool = False

    @classmethod
    def from_pen_frame(cls, frame: "object") -> "CoordSample":
        """``core.PenFrame``에서 CSV 샘플을 만든다."""
        raw = frame.raw_fingertip
        filt = frame.fingertip
        return cls(
            hand_detected=frame.hand_detected,
            raw_x=raw[0] if raw is not None else None,
            raw_y=raw[1] if raw is not None else None,
            raw_z=frame.raw_z,
            filtered_x=filt[0] if filt is not None else None,
            filtered_y=filt[1] if filt is not None else None,
            pen_ratio=frame.pen_ratio,
            pen_down=frame.pen_down,
        )


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else ""


class CoordRecorder:
    """좌표 CSV 기록기. ``start()``로 새 파일을 열고 ``write()``로 프레임을 남긴다.

    ``recording``이 False일 때 ``write()``는 아무 일도 하지 않으므로, 호출부는 기록 여부를
    검사할 필요 없이 매 프레임 ``write()``를 호출하면 된다.
    """

    def __init__(self, output_dir: str | Path = "output") -> None:
        self._output_dir = Path(output_dir)
        self._file = None
        self._writer: csv.writer | None = None
        self._frame_id = 0
        self._path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._file is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self) -> Path:
        """새 CSV 파일을 열고 헤더를 쓴다. 이미 기록 중이면 기존 파일을 닫고 새로 시작."""
        self.stop()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._output_dir / f"coords_{ts}.csv"
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._frame_id = 0
        return self._path

    def write(self, sample: CoordSample, *, timestamp: float | None = None) -> None:
        """기록 중일 때만 한 프레임을 CSV에 남긴다."""
        if self._writer is None or self._file is None:
            return
        t = time.time() if timestamp is None else timestamp
        self._writer.writerow([
            self._frame_id,
            f"{t:.6f}",
            int(sample.hand_detected),
            _fmt(sample.raw_x, ".2f"),
            _fmt(sample.raw_y, ".2f"),
            _fmt(sample.raw_z, ".5f"),
            _fmt(sample.filtered_x, ".2f"),
            _fmt(sample.filtered_y, ".2f"),
            _fmt(sample.pen_ratio, ".4f"),
            int(sample.pen_down),
        ])
        if self._frame_id % _FLUSH_EVERY == 0:
            self._file.flush()
        self._frame_id += 1

    def stop(self) -> None:
        """현재 파일을 닫는다. 기록 중이 아니면 무시."""
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    def toggle(self) -> bool:
        """기록 상태를 전환하고 기록 중이면 True를 반환."""
        if self.recording:
            self.stop()
            return False
        self.start()
        return True

    def close(self) -> None:
        self.stop()
