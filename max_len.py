import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像（可保留）

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from tqdm import tqdm


# =========================
# 1) 配置
# =========================
class Config:
    model_name = 'bert-base-chinese'
    train_path = './data/THUCNews-txt/train.txt'
    dev_path   = './data/THUCNews-txt/dev.txt'
    test_path  = './data/THUCNews-txt/test.txt'

    class_list = ['财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']
    label2id = {label: i for i, label in enumerate(class_list)}
    id2label = {i: label for i, label in enumerate(class_list)}
    num_classes = len(class_list)

    batch_size = 32
    epochs = 3
    learning_rate = 2e-5
    warmup_ratio = 0.0  # 你也可以改成 0.1 看看是否更稳
    max_grad_norm = 1.0

    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验设置：小消融（max_len 两档）
    ablation_max_lens = [128, 256]

    # 训练规范：按 Dev Macro-F1 保存最优
    save_metric = "macro_f1"  # 可选: "acc"
    early_stopping = True
    patience = 2  # 连续 patience 个 epoch 不提升则停止

    # 输出目录
    output_dir = "./outputs_bert"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 2) 数据读取
# =========================
def load_data(file_path, max_samples=None):
    """
    支持：
      - Text \t 3
      - 教育 \t 一段文本...
      - 3 \t 一段文本...
      - 一段文本... \t 教育
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_samples and len(lines) > max_samples:
        lines = random.sample(lines, max_samples)

    texts, labels = [], []
    success_count = 0

    for line in tqdm(lines, desc="Parsing"):
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 2:
            parts = line.split()
        if len(parts) < 2:
            continue

        p1 = parts[0].strip()
        p2 = parts[-1].strip()

        content = ""
        label_id = -1

        if p2.isdigit():  # A: label in back as digit
            label_id = int(p2)
            content = p1
        elif p1 in Config.label2id:  # B: label in front as Chinese
            label_id = Config.label2id[p1]
            content = p2
        elif p1.isdigit():  # C: label in front as digit
            label_id = int(p1)
            content = p2
        elif p2 in Config.label2id:  # D: label in back as Chinese
            label_id = Config.label2id[p2]
            content = p1

        if 0 <= label_id < Config.num_classes and content:
            texts.append(content)
            labels.append(label_id)
            success_count += 1

    print(f"  -> 成功读取: {success_count} 条数据")
    if success_count == 0:
        raise ValueError("未读取到任何有效数据，请检查数据格式或 class_list 是否一致。")
    return texts, labels


# =========================
# 3) Dataset（额外返回 raw_text，便于误判导出）
# =========================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            "raw_text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# =========================
# 4) 评估：Acc + Macro-F1 + probs + 文本
# =========================
@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()

    all_true, all_pred = [], []
    all_conf, all_text = [], []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=mask)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        all_true.extend(targets.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())
        all_text.extend(batch["raw_text"])  # list of str

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro")

    return acc, macro_f1, all_true, all_pred, all_conf, all_text


# =========================
# 5) 可视化：训练曲线
# =========================
def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["val_acc"], marker="o")
    plt.title("Dev Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["val_macro_f1"], marker="o")
    plt.title("Dev Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] 训练曲线已保存: {save_path}")


# =========================
# 6) 可视化：混淆矩阵
# =========================
def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] 混淆矩阵已保存: {save_path}")


# =========================
# 7) 导出误判样本（Top-K，按置信度降序）
# =========================
def export_top_misclassified(y_true, y_pred, conf, texts, save_path, top_k=50):
    rows = []
    for t, p, c, txt in zip(y_true, y_pred, conf, texts):
        if t != p:
            rows.append({
                "true_label": Config.id2label[t],
                "pred_label": Config.id2label[p],
                "confidence": float(c),
                "text": txt
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[WARN] 没有误判样本，未生成误判文件。")
        return

    df = df.sort_values(by="confidence", ascending=False).head(top_k)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Top误判样本已导出: {save_path}  (top_k={top_k})")


# =========================
# 8) 训练（记录 Dev Acc + Macro-F1；保存 best checkpoint）
# =========================
def train_model(model, train_loader, dev_loader, optimizer, scheduler, device, save_dir):
    history = {"train_loss": [], "val_acc": [], "val_macro_f1": []}

    best_score = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    best_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(Config.epochs):
        print(f"\n======== Epoch {epoch + 1}/{Config.epochs} ========")
        model.train()
        total_loss = 0.0

        start_time = time.time()
        progress = tqdm(train_loader, desc="Training", leave=False)

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=targets)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_train_loss)

        # Dev 评估
        val_acc, val_f1, _, _, _, _ = evaluate_model(model, dev_loader, device)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        epoch_time = time.time() - start_time
        print(f"Train Loss: {avg_train_loss:.4f} | Dev Acc: {val_acc:.4f} | Dev Macro-F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")

        # 保存最优模型（按 Dev Macro-F1 或 Acc）
        score = val_f1 if Config.save_metric == "macro_f1" else val_acc
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"[OK] 保存 best checkpoint: {best_path} | best_{Config.save_metric}={best_score:.4f} (epoch {best_epoch})")
        else:
            no_improve_epochs += 1
            if Config.early_stopping and no_improve_epochs >= Config.patience:
                print(f"[EARLY STOP] 连续 {Config.patience} 个 epoch 无提升，提前停止。best epoch = {best_epoch}")
                break

    return history, best_path, best_epoch


# =========================
# 9) 单次实验（给定 max_len）
# =========================
def run_one_experiment(max_len: int):
    print("\n" + "="*80)
    print(f"Running experiment: max_len = {max_len}")
    print("="*80)

    exp_dir = os.path.join(Config.output_dir, f"maxlen_{max_len}")
    os.makedirs(exp_dir, exist_ok=True)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.model_name)

    # data
    train_texts, train_labels = load_data(Config.train_path, max_samples=None)
    dev_texts, dev_labels     = load_data(Config.dev_path, max_samples=None)
    test_texts, test_labels   = load_data(Config.test_path, max_samples=10000)

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len)
    dev_dataset   = NewsDataset(dev_texts, dev_labels, tokenizer, max_len)
    test_dataset  = NewsDataset(test_texts, test_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # model
    model = BertForSequenceClassification.from_pretrained(
        Config.model_name,
        num_labels=Config.num_classes
    ).to(Config.device)

    # optim & scheduler
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    total_steps = len(train_loader) * Config.epochs
    warmup_steps = int(Config.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # train
    train_start = time.time()
    history, best_path, best_epoch = train_model(
        model, train_loader, dev_loader,
        optimizer, scheduler,
        Config.device, exp_dir
    )
    train_time = time.time() - train_start

    # curves
    plot_history(history, os.path.join(exp_dir, "training_curves.png"))

    # load best checkpoint for final test
    model.load_state_dict(torch.load(best_path, map_location=Config.device))

    # final test (ONLY ONCE)
    test_start = time.time()
    test_acc, test_f1, y_true, y_pred, conf, texts = evaluate_model(model, test_loader, Config.device)
    test_time = time.time() - test_start

    print("\n========== Final Evaluation on Test Set ==========")
    print(f"Best epoch (by dev {Config.save_metric}): {best_epoch}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=Config.class_list, digits=4))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        Config.class_list,
        title=f"BERT Confusion Matrix (max_len={max_len})",
        save_path=os.path.join(exp_dir, "confusion_matrix.png")
    )

    # export misclassified
    export_top_misclassified(
        y_true, y_pred, conf, texts,
        save_path=os.path.join(exp_dir, "top_misclassified.csv"),
        top_k=50
    )

    # save summary
    summary = {
        "max_len": max_len,
        "best_epoch": best_epoch,
        "dev_best_metric": Config.save_metric,
        "dev_best_value": max(history["val_macro_f1"]) if Config.save_metric == "macro_f1" else max(history["val_acc"]),
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "train_time_sec": train_time,
        "test_time_sec": test_time
    }
    with open(os.path.join(exp_dir, "summary.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] 本次实验输出目录: {exp_dir}")
    return summary


# =========================
# 10) 主程序：自动做 max_len 消融
# =========================
def main():
    set_seed(Config.seed)
    os.makedirs(Config.output_dir, exist_ok=True)

    print(f"Using Device: {Config.device}")
    print(f"Save metric: dev {Config.save_metric}")
    print(f"Early stopping: {Config.early_stopping} (patience={Config.patience})")

    all_summaries = []
    for max_len in Config.ablation_max_lens:
        summary = run_one_experiment(max_len)
        all_summaries.append(summary)

    df = pd.DataFrame(all_summaries)
    df = df.sort_values(by="test_macro_f1", ascending=False)
    out_csv = os.path.join(Config.output_dir, "ablation_summary.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n================ Ablation Summary ================\n")
    print(df.to_string(index=False))
    print(f"\n[OK] 消融汇总表已保存: {out_csv}")
    print("\n你可以在论文里直接引用该表，并讨论 max_len 对性能与成本的影响。")


if __name__ == "__main__":
    main()
