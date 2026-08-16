"""
LLM 교정 모듈

역할:
- ocr_infer.py에서 인식한 텍스트를 입력값으로 받아 gemma 모델로 문법 교정을 한다.
- ask() 함수 로직은 원본 그대로 두고, question 값만
  input() 대신 OCR 결과를 받아오도록 바꿨다.

사용법 (단독 실행, 전체 파이프라인 실행):
    python llm_correct.py 이미지경로.jpg
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ocr_infer import run_ocr, run_ocr_folder


BASE_MODEL = "google/gemma-4-E2B-it"


SYSTEM_PROMPT = '''
전달된 텍스트는 손글씨를 OCR로 인식한 결과라서, 인식 오류 때문에
실제로는 존재하지 않는 단어나 오타가 섞여 있을 수 있어.
 
규칙:
1. 실제로 존재하지 않는 단어(오타, OCR 인식 오류로 보이는 단어)가 있으면,
   생김새(철자)가 가장 비슷한 실제 단어로 복원해줘.
2. 조사, 어미 같은 문법적으로 어색한 부분이 있으면 자연스럽게 고쳐줘.
3. 이미 올바른 단어나 문장은 그대로 둬.
   의미를 바꾸는 재작성, 다른 뜻의 단어로 교체, 문장 추가는 하지 마.
4. 입력이 한글이면 한글로, 영어면 영어로 결과를 출력해줘.
5. 결과는 교정된 텍스트만 출력해. 설명, 이유, 따옴표, 그 외의 말은
   절대 덧붙이지 마.
 
예시:
어파이프 -> 미디어파이프
allevicte -> alleviate
적겪 -> 적격
Zcbla Is -> Zebra Is
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
).to(device)

model.eval()


def ask(question: str) -> str:

    # Gemma는 별도의 system role을 안정적으로 지원하지 않는 경우가 많아서,
    # 지시문(SYSTEM_PROMPT)을 user 메시지 안에 같이 넣어준다.
    prompt = f"{SYSTEM_PROMPT.strip()}\n\n입력: {question}\n출력:"

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        output,
        skip_special_tokens=True
    )

    return answer.strip()


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # 폴더 경로를 넣으면 폴더 전체를, 이미지 경로를 넣으면 한 장만 처리한다.
    target = sys.argv[1] if len(sys.argv) > 1 else "./Img"

    final_results = {}

    if Path(target).is_dir():
        ocr_results = run_ocr_folder(target)

        for i, (path, question) in enumerate(ocr_results.items(), 1):
            print(f"\n[{i}/{len(ocr_results)}] 교정 중: {path}")
            print(f"OCR 결과   : {question}")

            answer = ask(question) if question else ""
            print(f"교정 결과  : {answer}")

            final_results[path] = {
                "ocr_text": question,
                "corrected_text": answer,
            }

        out_path = "corrected_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n전체 결과 저장 완료: {out_path}")

    else:
        question = run_ocr(target)

        print("OCR 결과 (LLM 입력값)")
        print(question)
        print("=" * 60)

        answer = ask(question)

        print("\n교정 결과")
        print(answer)
        print("=" * 60)