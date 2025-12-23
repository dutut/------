import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 添加到代码开头，使用镜像
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条库，让控制台看起来更专业

# 1. 配置参数 (Configuration)
class Config:
    model_name = 'bert-base-chinese'
    train_path = './data/THUCNews-txt/train.txt'
    dev_path   = './data/THUCNews-txt/dev.txt'
    test_path  = './data/THUCNews-txt/test.txt'
    # 类别映射 (根据THUCNews的标准类别)
    class_list = ['财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']
    label2id = {label: i for i, label in enumerate(class_list)}
    id2label = {i: label for i, label in enumerate(class_list)}
    num_classes = len(class_list)
    
    max_len = 128         # 文本截断长度，显存够可调至256或512
    batch_size = 32       # 显存如果不够（如OOM），请调小至 16 或 8
    epochs = 3
    learning_rate = 2e-5  # BERT微调的经典学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.seed)

# 2. 数据处理 (Real Data Loading)
def load_data(file_path, max_samples=None):
    """
    读取 cnews 格式数据: Label <tab> Text  或 Text <tab> Label
    支持：
        - 文本在前，数字标签在后： Text \t 3
        - 标签在前，文本在后：     教育 \t 一段文本...
        - 标签为中文或者数字均可
    max_samples: 用于调试，如果数据太大可以限制读取数量
    """
    print(f"Loading data from {file_path}...")
    texts = []
    labels = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if max_samples and len(lines) > max_samples:
        lines = random.sample(lines, max_samples)
    
    success_count = 0
    
    for line in tqdm(lines):
        line = line.strip() 
        if not line:
            continue
        
        # 按制表符分割
        parts = line.split('\t')
        
        # 如果没有tab，尝试用空格分割（兼容性处理）
        if len(parts) < 2:
            parts = line.split()
            
        if len(parts) < 2:
            continue  # 实在分不开，跳过
            
        # === 自动判断标签位置 ===
        p1 = parts[0].strip()   # 第一部分
        p2 = parts[-1].strip()  # 最后一部分
        
        content = ""
        label_id = -1
        
        # 情况 A：标签在后面，且是数字 (如： "一段文本 \t 3")
        if p2.isdigit():
            label_id = int(p2)
            content = p1
            
        # 情况 B：标签在前面，且是中文 (如： "教育 \t 一段文本")
        elif p1 in Config.label2id:
            label_id = Config.label2id[p1]
            content = p2
            
        # 情况 C：标签在前面，但已经是数字了
        elif p1.isdigit():
            label_id = int(p1)
            content = p2
            
        # 情况 D：标签在后面，且是中文
        elif p2 in Config.label2id:
            label_id = Config.label2id[p2]
            content = p1
            
        # === 保存数据 ===
        # 确保解析出的 label_id 在有效范围内 (0 ~ num_classes-1)
        if 0 <= label_id < Config.num_classes and content:
            texts.append(content)
            labels.append(label_id)
            success_count += 1
            
    print(f"  -> 成功读取: {success_count} 条数据")
    
    if success_count == 0:
        raise ValueError("未读取到任何有效数据，请检查数据格式或 Config.class_list 是否与标签对应。")
            
    return texts, labels


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(  # 将原始文本一次性编码为模型可用的输入（ID、mask 等）
            text,  # 待编码的文本内容（这里确保转成 str，避免出现非字符串导致的类型错误）
            add_special_tokens=True,  # 自动添加模型需要的特殊符号：如 BERT 的 [CLS] 开头、[SEP] 结尾（不同模型规则不同）
            max_length=self.max_len,  # 统一序列长度上限：超过则截断、不足则补齐，保证 batch 内张量形状一致
            padding='max_length',  # 按 max_length 补齐：短文本会在末尾用 pad_token 补到固定长度
            truncation=True,  # 开启截断：长文本会按 tokenizer 的策略截到 max_length
            return_token_type_ids=False,  # 不返回 token_type_ids（segment ids）：单句分类一般用不到，少返回一个张量更省显存/带宽
            return_attention_mask=True,  # 返回 attention_mask：1 表示有效 token，0 表示 padding，供模型忽略补齐部分
            return_tensors='pt',  # 返回 PyTorch 张量（torch.Tensor），便于后续直接送入模型
        )  # encode_plus 返回一个 dict-like 对象：常见键包括 input_ids、attention_mask（以及可选的 token_type_ids）
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. 训练与评估流程
def train_model(model, train_loader, val_loader, optimizer, scheduler, device):
    """
    train_loader: 只包含训练集
    val_loader:   只包含验证集（dev），用于监控收敛，不参与参数更新
    ❗ 测试集（test）不会在本函数中出现，避免信息泄漏
    """
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(Config.epochs):
        print(f'\n======== Epoch {epoch + 1}/{Config.epochs} ========')
        model.train()
        total_loss = 0.0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=targets)
            loss = outputs.loss
            
            total_loss += loss.item()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # ====== 在验证集（dev）上评估，不用测试集 ======
        val_acc, _ = evaluate_model(model, val_loader, device)
        history['val_acc'].append(val_acc)
        
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Accuracy (Dev): {val_acc:.4f}')
        
    return history


def evaluate_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=targets)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            real_values.extend(targets.cpu().tolist())
            
    return accuracy_score(real_values, predictions), (real_values, predictions)

# 4. 可视化函数 (Visualization)
def plot_history(history):
    """
    绘制 Loss 下降曲线和 Accuracy 上升曲线
    使用的 accuracy 是验证集（dev）的结果
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'o-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制 Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'o-', label='Validation Accuracy (Dev)')
    plt.title('Validation Accuracy on Dev Set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_result.png')  # 保存图片到本地
    print("图表已保存为 training_result.png")
    plt.show()

# 5. 主程序入口
if __name__ == '__main__':
    print(f"Using Device: {Config.device}")
    
    # 1. 准备分词器
    tokenizer = BertTokenizer.from_pretrained(Config.model_name)
    
    # 2. 准备数据
    # 训练集（train）：用于参数学习
    train_texts, train_labels = load_data(Config.train_path, max_samples=None)
    
    # 验证集（dev）：用于调参 / 看收敛 / 画曲线
    dev_texts, dev_labels = load_data(Config.dev_path, max_samples=None)
    
    # 测试集（test）：只在整个训练完全结束后评估一次
    # 可以限制 max_samples 以缩短运行时间，也可以设为 None 使用全部
    test_texts, test_labels = load_data(Config.test_path, max_samples=10000)
    
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, Config.max_len)
    dev_dataset   = NewsDataset(dev_texts, dev_labels, tokenizer, Config.max_len)
    test_dataset  = NewsDataset(test_texts, test_labels, tokenizer, Config.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=Config.batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=Config.batch_size)
    
    # 3. 初始化模型
    print("Initializing BERT Model...")
    model = BertForSequenceClassification.from_pretrained(
        Config.model_name,
        num_labels=Config.num_classes
    )
    model = model.to(Config.device)
    
    # 4. 优化器与学习率调度器
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    total_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 5. 在 train + dev 上进行训练与验证（不会使用 test）
    history = train_model(model, train_loader, dev_loader, optimizer, scheduler, Config.device)
    
    # 6. 训练过程可视化（train loss + dev accuracy）
    plot_history(history)
    
    # 7. 最终评估报告（只在测试集 test 上评估一次，避免信息泄漏）
    print("\n========== Final Evaluation on Test Set ==========")
    _, (y_true, y_pred) = evaluate_model(model, test_loader, Config.device)
    
    print(classification_report(
        y_true,
        y_pred,
        target_names=Config.class_list,
        digits=4  # 保留4位小数，看起来更精确
    ))
