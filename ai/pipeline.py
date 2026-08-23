"""controller.ocr_llm_pipeline.OcrLlmPipelineWorker에 꽂아 넣는 실제 OCR/LLM 함수.

``ocr_stage``/``llm_stage``는 그 워커가 기대하는 시그니처
(``ocr_fn(image: np.ndarray) -> str``, ``llm_fn(raw_text: str) -> str``)를 그대로 따른다.
무거운 torch/transformers 모델 로딩(``ai.llm_correct`` 모듈 최상단)은 ``llm_stage``가
처음 호출될 때만 일어나도록 지연 임포트한다 — 세션 시작 시 바로 로딩되면 앱 시작이
느려지고, 어차피 이 함수는 백그라운드 워커 스레드에서만 호출되므로 지연시켜도 메인
루프를 막지 않는다.
"""

from __future__ import annotations

import numpy as np


def _yield_cpu_to_main_thread() -> None:
    """torch가 코어를 전부 차지하지 않도록 이 백그라운드 스레드에서만 스레드 수를 낮춘다.

    ``torch.set_num_threads``는 프로세스 전역 설정이지만, torch를 실제로 쓰는 곳이 이
    OCR/LLM 백그라운드 워커뿐이라(메인 루프는 MediaPipe/OpenCV만 씀) 부작용이 없다.
    실측 기준 이게 없으면 메인 스레드의 MediaPipe 추적이 코어를 못 받아 프레임당
    11.81ms -> 18.99ms(최대 459ms 스파이크)까지 느려졌다.
    """
    import torch

    torch.set_num_threads(1)


def ocr_stage(image: np.ndarray) -> str:
    """캔버스 스냅샷(ndarray)에 바로 OCR을 돌린다 (디스크 왕복 없음)."""
    _yield_cpu_to_main_thread()
    from ai.ocr_infer import run_ocr_array

    return run_ocr_array(image)


def llm_stage(raw_text: str) -> str:
    """OCR 결과 문자열을 로컬 LLM(gemma)으로 오탈자 교정한다."""
    if not raw_text.strip():
        return ""
    _yield_cpu_to_main_thread()
    from ai.llm_correct import ask

    return ask(raw_text)
