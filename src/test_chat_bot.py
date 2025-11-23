import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def get_base_dir():
    if os.environ.get("CHECKPOINT_DIR"):
        return os.environ.get("CHECKPOINT_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        return os.path.join(cache_dir, "fine-tuned_weights")

CHECKPOINT_DIR = get_base_dir()
# Ensure this matches the logic in fine_tune.py
ADAPTER_PATH = os.environ.get("OUTPUT_DIR", CHECKPOINT_DIR)
BASE_MODEL = "ministral/Ministral-3b-instruct"

print(f"Loading adapters from: {ADAPTER_PATH}")

# Load Tokenizer
# We try to load from the adapter path first, as fine_tune.py saves it there.
try:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
except OSError:
    print(f"Tokenizer not found at {ADAPTER_PATH}, loading from base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit Quantization Config (Matching fine_tune.py)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa" # Optimized for RTX 4090
)

# Load LoRA Adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def chat(query, max_new_tokens=256):
    messages = [{"role": "user", "content": query}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    generated_ids = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    print("\nBot ready! Type 'quit' or 'exit' to stop.")
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue
                
            response = chat(user_input)
            print(f"Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
