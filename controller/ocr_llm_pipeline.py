"""OCR+LLM 비동기 파이프라인 스캐폴딩 — 실제 모델 로직은 플레이스홀더.

C(소연/민진)가 EasyOCR 파라미터 튜닝·전처리와 Ollama 프롬프트를 별도로 만들고 있어,
여기서는 트리거 -> 비동기 처리 -> 상태/결과 노출까지의 배관만 완성한다.
``placeholder_ocr``/``placeholder_llm``은 나중에 ``ai.ocr``/``ai.llm``의 실제 호출로
교체될 자리다 (TODO(C) 표시). ``ai/`` 디렉터리는 C 담당이라 여기서 임포트하지 않는다.

``OcrLlmPipelineWorker.submit``/``poll``은 항상 단일 호출자(OpenCV 데모 루프 스레드
또는 PyQt GUI 스레드 중 하나)에서만 호출된다는 전제로 락 없이 구현했다 — 다른 스레드에서
동시에 호출하면 안전하지 않다.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

__all__ = [
    "STATUS_IDLE",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "STATUS_ERROR",
    "PipelineResult",
    "LlmStageError",
    "placeholder_ocr",
    "placeholder_llm",
    "OcrLlmPipelineWorker",
]

STATUS_IDLE = "IDLE"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_ERROR = "ERROR"

DEFAULT_TIMEOUT_SEC = 120.0


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """``OcrLlmPipelineWorker.poll()``이 매 프레임 내놓는 상태 스냅샷."""

    status: str  # STATUS_IDLE | STATUS_RUNNING | STATUS_DONE | STATUS_ERROR
    ocr_text: str | None
    corrected_text: str | None
    error: str | None


class LlmStageError(RuntimeError):
    """``llm_fn``이 실패했을 때, 이미 성공한 ``ocr_text``를 실어 나르기 위한 예외.

    OCR 자체는 성공했는데 그 다음 LLM 교정만 실패한 경우, 일반 예외로 처리하면
    ``poll()``이 ``ocr_text``까지 통째로 버리게 된다. 이 예외는 ``ocr_text``를
    속성으로 들고 있어 ``poll()``이 ERROR 상태에서도 그 텍스트를 보존할 수 있게 한다.
    """

    def __init__(self, ocr_text: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.ocr_text = ocr_text


def placeholder_ocr(image: np.ndarray) -> str:
    """TODO(C): ``ai.ocr``의 실제 EasyOCR 호출로 교체.

    스냅샷이 제대로 들어왔는지 스모크 테스트에서 바로 확인할 수 있도록 이미지 크기를
    더미 결과에 포함시킨다.
    """
    time.sleep(0.3)
    height, width = image.shape[:2]
    return f"(placeholder OCR: {width}x{height} image)"


def placeholder_llm(raw_text: str) -> str:
    """TODO(C): ``ai.llm``의 실제 Ollama(localhost:11434) 호출로 교체."""
    time.sleep(0.3)
    return f"(placeholder LLM correction of: {raw_text})"


class OcrLlmPipelineWorker:
    """OCR->LLM 두 단계를 백그라운드 스레드 하나에서 순서대로 실행하는 워커.

    ``max_workers=1``은 "동시에 스레드 하나만 띄운다"는 뜻일 뿐이고, 실제 "한 번에
    작업 하나만" 보장은 ``submit()``의 ``is_running`` 게이트가 한다.
    """

    def __init__(
        self,
        *,
        ocr_fn=placeholder_ocr,
        llm_fn=placeholder_llm,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-llm")
        self._ocr_fn = ocr_fn
        self._llm_fn = llm_fn
        self._timeout_sec = timeout_sec
        self._future: Future | None = None
        self._submitted_at: float | None = None
        self._last_result = PipelineResult(STATUS_IDLE, None, None, None)

    @property
    def is_running(self) -> bool:
        return self._future is not None and not self._future.done()

    def submit(self, image: np.ndarray) -> bool:
        """이미지 스냅샷 하나를 백그라운드로 처리 요청한다.

        이미 처리 중이면 아무것도 하지 않고 ``False``를 반환한다(한 번에 한 작업만).
        """
        if self.is_running:
            return False
        self._future = self._executor.submit(self._run, image)
        self._submitted_at = time.monotonic()
        return True

    def _run(self, image: np.ndarray) -> tuple[str, str]:
        ocr_text = self._ocr_fn(image)
        try:
            corrected_text = self._llm_fn(ocr_text)
        except Exception as exc:  # noqa: BLE001 — ocr_text를 실어 보존하기 위해 감싼다.
            raise LlmStageError(ocr_text, exc) from exc
        return ocr_text, corrected_text

    def poll(self) -> PipelineResult:
        """호출 스레드를 절대 블로킹하지 않는다. 프레임마다 한 번씩 부를 것."""
        future = self._future
        if future is None:
            return self._last_result
        if not future.done():
            if (
                self._submitted_at is not None
                and time.monotonic() - self._submitted_at > self._timeout_sec
            ):
                # generate()가 멈춘 스레드는 안전하게 강제 종료할 수 없다 — 추적을
                # 포기하고 새 실행기로 교체해, 그 스레드는 백그라운드에서 홀로 계속
                # 돌게 두고(자원은 소모하지만) 이후 트리거는 즉시 다시 가능하게 한다.
                self._last_result = PipelineResult(
                    STATUS_ERROR,
                    None,
                    None,
                    f"시간 초과({self._timeout_sec:.0f}초) — 이전 작업은 백그라운드에서 계속 실행됩니다.",
                )
                self._future = None
                self._submitted_at = None
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-llm")
                return self._last_result
            self._last_result = PipelineResult(STATUS_RUNNING, None, None, None)
            return self._last_result

        self._future = None  # 정확히 한 번만 소비.
        self._submitted_at = None
        try:
            ocr_text, corrected_text = future.result()
        except LlmStageError as exc:
            self._last_result = PipelineResult(STATUS_ERROR, exc.ocr_text, None, str(exc))
        except Exception as exc:  # noqa: BLE001 — 예외 종류를 가리지 않고 상태로 변환한다.
            self._last_result = PipelineResult(STATUS_ERROR, None, None, str(exc))
        else:
            self._last_result = PipelineResult(STATUS_DONE, ocr_text, corrected_text, None)
        return self._last_result

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
