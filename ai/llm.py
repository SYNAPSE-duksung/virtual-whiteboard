import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

#실행은 아직 안하고, 코드만 작성"
BASE_MODEL = "google/gemma-4-E2B-it"


SYSTEM_PROMPT = '''
전달된 텍스트를 문법적 오류 없이 교정해야해.
입력된 언어가 한글이면 한글로, 영어이면 영어로 교정해야해.
만약 한글에 영어나 기호가 이상하게 섞여있다면, 문맥이나 의미를 고려해서 한글로 바꿔줘야해.
그리고 교정을 하고 나서 나에게 알려줄 때, 다른 말은 덧붙이지 말고 내가 보낸 텍스트를 교정한 것만 보내줘야해.
말을 덧붙이면 안돼.
예시: [[어파이프 -> 미디어파이프

'''
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)


model.eval()


print(model.peft_config)
for name, param in model.named_parameters():
    if "lora" in name.lower():
        print(name, param.shape, param.abs().mean().item())
        break
def ask(question):

    messages = [

        {
            "role": "user",
            "content": question
        }
    ]
    print(type(question))
    print(repr(question))

    print(messages)
    print(type(messages))
    print(type(messages[0]["content"]))
    inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,   # 추가
    ).to(model.device)
    print("=== 실제 모델에 들어가는 프롬프트 ===")
    print(tokenizer.decode(inputs["input_ids"][0]))
    print("=" * 60)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,   # 수정
            max_new_tokens=256,
            temperature=0.2,    #수정 전에는 0.7이었음
            top_p=0.9,
            do_sample=True, #수정 전에는 True였음
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 입력 부분 제외
    output = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        output,
        skip_special_tokens=True
    )

    return answer.strip()

while True:

    question = input("질문 : ")

    if question.lower() == "exit":
        break

    answer = ask(question)

    print("\n답변")
    print(answer)
    print("=" * 60)