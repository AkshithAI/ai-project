from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ministral/Ministral-3b-instruct")
model = AutoModelForCausalLM.from_pretrained("ministral/Ministral-3b-instruct")