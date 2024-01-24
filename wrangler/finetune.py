import os
# Hack for my machine, and Windows
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

dialogs_dataset = load_dataset("dataset", data_files={"train": "train.txt", "test": "test.txt"})

model_name = "../models/OpenOrca-Platypus2-13B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

train_dataset = dialogs_dataset["train"].map(lambda x: tokenizer(x["text"]), batched=True)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        output_dir="OpenOrca-Platypus2-13B-finetuned",
        per_device_train_batch_size=4,
        warmup_steps=100,
        max_steps=20000,
        save_steps=400,
        learning_rate=1e-4,
        logging_steps=100,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

model.save_pretrained("OpenOrca-Platypus2-13B/peft-model")
