import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .models import tokenizer

BASE_MODEL = "mistralai/Mistral-3B-Instruct"
ADAPTER_PATH = "/Users/apple/Documents/ai-project/ckpts"  

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map="auto")
model.eval()

def chat(query, max_new_tokens=128):
    messages = [{"role": "user", "content": query}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  
        )

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)

    user_only = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    response = full_text[len(user_only):].strip()
    return response

print(chat("How do I track my order?"))
