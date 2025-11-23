import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import wandb

# Attempt to clear cache before starting
torch.cuda.empty_cache()

# Initialize wandb
run = wandb.init(
    entity="akshithmarepally-akai",
    project="ai-project",
    config={
        "architecture": "GPT",
        "dataset": "AkshithAI/amazon-support-mistral3b",
    }
)

def get_base_dir():
    if os.environ.get("CHECKPOINT_DIR"):
        return os.environ.get("CHECKPOINT_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        return os.path.join(cache_dir, "fine-tuned_weights")

CHECKPOINT_DIR = get_base_dir()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", CHECKPOINT_DIR)

# --- STABLE HYPERPARAMETERS ---
NUM_EPOCHS = 1
# Reduced batch size to 8 to fit comfortably in memory
PER_DEVICE_BATCH_SIZE = 8  
# Increased accumulation to 8 to keep effective batch size at 64 (8 * 8)
GRAD_ACCUM = 8              
LEARNING_RATE = 2e-4
MAX_STEPS = 1000            
SAVE_STRATEGY = "steps"
SAVE_STEPS = 200
LOGGING_STEPS = 10
SEED = 42
MAX_SEQ_LENGTH = 512

# LoRA specific
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# Load Tokenizer
model_id = "ministral/Ministral-3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load and Process Dataset
ds = load_dataset("AkshithAI/amazon-support-mistral3b")

def process_data(samples):
    text_column = "text"
    if "text" not in samples and "content" in samples:
        text_column = "content"
    elif "text" not in samples and "prompt" in samples and "response" in samples:
        return tokenizer([p + " " + r for p, r in zip(samples["prompt"], samples["response"])], 
                         truncation=True, max_length=MAX_SEQ_LENGTH)

    if text_column in samples:
        return tokenizer(samples[text_column], truncation=True, max_length=MAX_SEQ_LENGTH)
    else:
        if "input_ids" in samples:
             return {"input_ids": [x[:MAX_SEQ_LENGTH] for x in samples["input_ids"]],
                     "attention_mask": [x[:MAX_SEQ_LENGTH] for x in samples["attention_mask"]]}
        raise ValueError(f"Could not find text column in dataset. Columns: {list(samples.keys())}")

tokenized_ds = ds.map(
    process_data,
    batched=True,
    remove_columns=ds["train"].column_names,
    desc="Tokenizing dataset"
)

train_ds = tokenized_ds["train"]
eval_ds = None

# bitsandbytes config for 4-bit load (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    fp16=False,
    logging_steps=LOGGING_STEPS,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    eval_strategy="no",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="wandb",
    dataloader_num_workers=4
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save fine-tuned weights
os.makedirs(OUTPUT_DIR, exist_ok=True)
artifact = wandb.Artifact("LoRA-adapters", type="model")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
artifact.add_dir(OUTPUT_DIR)
run.log_artifact(artifact)
run.finish()
print("LoRA adapters saved to:", OUTPUT_DIR)
