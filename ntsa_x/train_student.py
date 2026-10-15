"""Opt-in supervised LoRA student training from approved teacher samples.

The local student is independent of the teacher at training/inference time.
This trains weights, unlike recall/memoization. Requires optional packages:
torch, transformers, peft.
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import json
import math
import random


def load_data(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            turns = item["messages"]
            if (not isinstance(turns, list) or len(turns) < 2
                    or turns[-2]["role"] != "user"
                    or turns[-1]["role"] != "assistant"):
                raise ValueError("Expected validated conversation ending in user/assistant")
            rows.append(turns)
    if not rows:
        raise ValueError("Empty student dataset")
    return rows


def build_tokens(tokenizer, conversations, max_length):
    prepared = []
    for dialogue in conversations:
        prompt = tokenizer.apply_chat_template(
            dialogue[:-1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            dialogue, tokenize=True, add_generation_prompt=False)
        if not isinstance(prompt, list) or not isinstance(full, list):
            raise ValueError("Unsupported chat tokenizer")
        if full[:len(prompt)] != prompt:
            raise ValueError("Tokenizer prompt prefix mismatch; use an instruct chat template")
        if len(full) > max_length:
            continue
        labels = [-100] * len(prompt) + full[len(prompt):]
        if all(value == -100 for value in labels):
            continue
        prepared.append((full, labels))
    if not prepared:
        raise ValueError("No suitable trainable examples fit max-length")
    return prepared


def evaluate(model, examples, torch):
    model.eval()
    losses = []
    with torch.no_grad():
        for ids, labels in examples:
            batch = torch.tensor([ids], dtype=torch.long, device=model.device)
            target = torch.tensor([labels], dtype=torch.long, device=model.device)
            loss = model(input_ids=batch, labels=target).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite evaluation loss")
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    parser = ArgumentParser(description="NTSA-X offline teacher-to-student LoRA fine-tuning")
    parser.add_argument("--base-model", required=True,
                        help="Path to a local, licensed HF-format causal model")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    if (not 1 <= args.epochs <= 20 or not 128 <= args.max_length <= 8192
            or not 1e-7 <= args.learning_rate <= 0.001
            or args.rank not in (4, 8, 16, 32)):
        parser.error("Invalid supervised fine-tuning hyperparameters")
    if Path(args.output).exists():
        parser.error("Refusing to overwrite student output")
    train_data, eval_data = load_data(args.train), load_data(args.eval)
    train_prompts = {row[-2]["content"] for row in train_data}
    eval_prompts = {row[-2]["content"] for row in eval_data}
    if train_prompts & eval_prompts:
        raise ValueError("Training and evaluation contain overlapping prompts")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install optional torch, transformers, and peft packages") from exc

    random.seed(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Base model must ship with an explicit chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples = build_tokens(tokenizer, train_data, args.max_length)
    eval_examples = build_tokens(tokenizer, eval_data, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate)
    before = evaluate(model, eval_examples, torch)
    completed_steps = 0
    for _epoch in range(args.epochs):
        random.shuffle(train_examples)
        model.train()
        for ids, labels in train_examples:
            batch = torch.tensor([ids], dtype=torch.long, device=model.device)
            target = torch.tensor([labels], dtype=torch.long, device=model.device)
            loss = model(input_ids=batch, labels=target).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
    after = evaluate(model, eval_examples, torch)
    output = Path(args.output)
    output.mkdir(parents=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    report = {
        "student_base_model": str(args.base_model),
        "train_examples": len(train_examples), "holdout_examples": len(eval_examples),
        "train_optimizer_steps": completed_steps, "before_eval_loss": before,
        "after_eval_loss": after, "after_eval_perplexity": math.exp(min(after, 50)),
        "teacher_calls_during_training": 0, "independent_generalization_proven": False,
        "requires_human_quality_evaluation": True,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
