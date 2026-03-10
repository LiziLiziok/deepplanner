"""
GPU占用脚本：使用transformers库加载模型并进行无限循环训练
用途：占用指定GPU，防止被其他任务抢占
使用方法：python runrunrun.py --gpu 0,1,2,3 --model_path /path/to/model
按Ctrl+C退出
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random


class RandomTextDataset(Dataset):
    """Generate random training data"""
    def __init__(self, tokenizer, num_samples=1000, max_length=512):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        # Pre-generate random token ids
        self.vocab_size = tokenizer.vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input_ids
        seq_len = random.randint(self.max_length // 2, self.max_length)
        input_ids = torch.randint(0, self.vocab_size, (seq_len,))
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def data_collator(features):
    """Collate function for padding"""
    max_len = max(len(f["input_ids"]) for f in features)
    
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        batch["input_ids"].append(
            torch.cat([f["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        batch["attention_mask"].append(
            torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        batch["labels"].append(
            torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
    
    return {k: torch.stack(v) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="GPU占用脚本")
    parser.add_argument("--gpu", type=str, default="0", help="要占用的GPU编号，多个用逗号分隔，如: 0,1,2,3")
    parser.add_argument("--model_path", type=str, 
                        default="/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
                        help="模型路径")
    parser.add_argument("--batch_size", type=int, default=4, help="训练batch size")
    parser.add_argument("--max_length", type=int, default=512, help="序列最大长度")
    parser.add_argument("--num_samples", type=int, default=10000, help="随机生成的样本数量")
    args = parser.parse_args()
    
    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu_count = len(args.gpu.split(","))
    
    print(f"=" * 60)
    print(f"G脚本启动")
    print(f"使用GPU: {args.gpu}")
    print(f"模型路径: {args.model_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"序列长度: {args.max_length}")
    print(f"按 Ctrl+C 退出")
    print(f"=" * 60)
    
    # Load tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("正在加载模型到GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to avoid gradient scaling issues
        device_map="auto" if gpu_count > 1 else "cuda:0",
        trust_remote_code=True
    )
    model.train()
    
    print(f"模型加载完成，占用GPU显存")
    
    # Create random dataset
    print("正在生成随机训练数据...")
    train_dataset = RandomTextDataset(tokenizer, num_samples=args.num_samples, max_length=args.max_length)
    
    # Training arguments - no saving
    training_args = TrainingArguments(
        output_dir="/tmp/gpu_occupy_temp",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=999999,  # Run forever
        logging_steps=100,
        save_strategy="no",  # Never save
        report_to="none",
        bf16=True,  # Use bf16 instead of fp16 to avoid gradient scaling issues
        dataloader_drop_last=True,
        remove_unused_columns=False,
        max_grad_norm=0.0,  # Disable gradient clipping to avoid unscale issues
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training loop
    print("开始无限循环训练...")
    print("GPU正在被占用，按 Ctrl+C 退出")
    
    try:
        while True:
            trainer.train()
            # Reset dataset for next round
            train_dataset = RandomTextDataset(tokenizer, num_samples=args.num_samples, max_length=args.max_length)
            trainer.train_dataset = train_dataset
    except KeyboardInterrupt:
        print("\n收到退出信号，正在退出...")
        print("GPU占用结束")


if __name__ == "__main__":
    main()

# python runrunrun.py --gpu 0,1 --batch_size 8 --max_length 1024 --num_samples 10000
