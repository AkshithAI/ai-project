from datasets import load_dataset
from .models import tokenizer
from huggingface_hub import HfApi, upload_folder
import os

ds = load_dataset("sentence-transformers/amazon-qa")

max_length = 512
def map_func(example):
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  
    )

    user_only_text = tokenizer.apply_chat_template(
        messages[:1],                 
        tokenize=False,
        add_generation_prompt=True,   
    )
    
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    user_only_tokens = tokenizer(
        user_only_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    user_len = len(user_only_tokens["input_ids"])

    labels = [-100] * len(input_ids)

    for i in range(user_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_ds = ds.map(
    map_func,
    remove_columns=ds["train"].column_names,  
)


tokenized_ds.save_to_disk("./tokenized_dataset")

HF_TOKEN = "SECRET_API_KEY"
REPO_ID = "AkshithAI/amazon-support-mistral3b"

api = HfApi()
try:
    api.create_repo(repo_id=REPO_ID, token=HF_TOKEN, repo_type="dataset")
except Exception as e:
    print("create_repo:", e)


upload_folder(
    folder_path="./tokenized_dataset",
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo="",   
    token=HF_TOKEN,
)
