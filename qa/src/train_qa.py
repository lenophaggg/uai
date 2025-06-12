# src/train_qa.py  (fixed for older args)
import json, os, argparse, multiprocessing as mp, torch
from pathlib import Path
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, default_data_collator)

# ────────── конфиг ───────────────────────────────────────────────────────
MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LEN    = 256
STRIDE     = 64
EPOCHS     = 2
LR         = 5e-4
CPU_CORES  = max(1, os.cpu_count() // 2)

torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(1)

# ────────── пути относительно корня проекта ──────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sberquad_faq.json"
OUT_DIR   = ROOT / "models" / "rubert-tiny2-faq"

# ────────── препроцессинг ────────────────────────────────────────────────
def preprocess(batch, tok):
    enc = tok(
        [q.strip() for q in batch["question"]],
        [c.strip() for c in batch["context"]],
        max_length=MAX_LEN, truncation="only_second",
        stride=STRIDE, padding="max_length",
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    sm, om = enc.pop("overflow_to_sample_mapping"), enc.pop("offset_mapping")
    s_pos, e_pos = [], []
    for i, offs in enumerate(om):
        idx   = sm[i]
        a_s   = batch["answers"][idx]["answer_start"][0]
        a_e   = a_s + len(batch["answers"][idx]["text"][0])
        seq   = enc.sequence_ids(i)
        try:
            c0  = seq.index(1)
            c1  = len(seq) - 1 - seq[::-1].index(1)
        except ValueError:
            s_pos.append(0), e_pos.append(0); continue
        if a_s < offs[c0][0] or a_e > offs[c1][1]:
            s_pos.append(0), e_pos.append(0); continue
        ts, te = c0, c1
        while ts <= c1 and offs[ts][0] <= a_s: ts += 1
        while te >= c0 and offs[te][1] >= a_e: te -= 1
        s_pos.append(max(c0, ts-1))
        e_pos.append(min(c1, te+1))
    enc["start_positions"], enc["end_positions"] = s_pos, e_pos
    return enc

# ────────── обучение ─────────────────────────────────────────────────────
def main(data_path: Path = DATA_PATH, out_dir: Path = OUT_DIR):
    data   = json.loads(data_path.read_text(encoding="utf-8"))
    split  = int(0.9 * len(data))
    ds_tr  = Dataset.from_list(data[:split])
    ds_val = Dataset.from_list(data[split:])

    tok    = AutoTokenizer.from_pretrained(MODEL_NAME)
    model  = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    for n, p in model.named_parameters():
        if not n.startswith("qa_outputs"):
            p.requires_grad_(False)

    ds_tr  = ds_tr.map(preprocess, fn_kwargs={"tok": tok},
                       batched=True, num_proc=CPU_CORES,
                       remove_columns=ds_tr.column_names)
    ds_val = ds_val.map(preprocess, fn_kwargs={"tok": tok},
                        batched=True, num_proc=CPU_CORES,
                        remove_columns=ds_val.column_names)

    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",      # ← старое имя параметра
        save_strategy="epoch",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        optim="adafactor",
        logging_steps=2000,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=ds_tr, eval_dataset=ds_val,
                      data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"✓ модель сохранена в {out_dir}")

# ────────── точка входа (Windows-safe) ───────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--out",  default=str(OUT_DIR))
    p = parser.parse_args()
    main(Path(p.data), Path(p.out))
