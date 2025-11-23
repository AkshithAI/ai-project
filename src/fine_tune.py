import os
from datasets import load_from_disk,load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from .models import tokenizer,model
import torch
import wandb

run = wandb.init(
    entity = "akshithmarepally-akai",
        project = "ai-project",
        config = {
            "architecture" : "GPT",
            "dataset" : "AkshithAI/amazon-support-mistral3b",
        }
)

ds = load_dataset("AkshithAI/amazon-support-mistral3b")

OUTPUT_DIR = "/Users/apple/Documents/ai-project/ckpts"
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4  
GRAD_ACCUM = 1
LEARNING_RATE = 3e-4
MAX_STEPS = -1             
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 50
SEED = 42
# LoRA specific
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]


train_ds = ds["train"]
eval_ds = None

# Tokenizer
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# bitsandbytes config for 4-bit load (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
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
    output_dir="./qllora_run",
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS if MAX_STEPS<=0 else 999999,
    max_steps=MAX_STEPS if MAX_STEPS>0 else -1,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=LOGGING_STEPS,
    save_strategy=SAVE_STRATEGY,
    evaluation_strategy="epoch" if eval_ds is not None else "no",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
)

data_collator = default_data_collator

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
artifact = wandb.Artifact("LoRA-adapters",type = "model")
model.save_pretrained(OUTPUT_DIR)
artifact.add_file(OUTPUT_DIR)
run.log_artifact(artifact)
run.finish()
print("LoRA adapters saved to:", OUTPUT_DIR)
