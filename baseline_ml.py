import os
import re
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
candidate_fonts = [
    "Noto Sans CJK JP",      
    "Noto Serif CJK JP",     
    "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans SC",
    "STHeiti", "Arial Unicode MS"
]
available = set(f.name for f in font_manager.fontManager.ttflist)
for f in candidate_fonts:
    if f in available:
        mpl.rcParams["font.sans-serif"] = [f]
        mpl.rcParams["font.family"] = "sans-serif"
        print(f"[OK] Matplotlib 使用中文字体: {f}")
        break
else:
    print("[WARN] 未找到候选中文字体")

mpl.rcParams["axes.unicode_minus"] = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# =========================
# 1. 配置
# =========================
class Config:
    train_path = './data/THUCNews-txt/train.txt'
    dev_path   = './data/THUCNews-txt/dev.txt'
    test_path  = './data/THUCNews-txt/test.txt'

    # 与你 BERT 代码保持一致
    class_list = ['财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']
    label2id = {label: i for i, label in enumerate(class_list)}
    id2label = {i: label for i, label in enumerate(class_list)}
    num_classes = len(class_list)

    seed = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


set_seed(Config.seed)

# =========================
# 2. 读取数据
# =========================
def load_data(file_path, max_samples=None):
    """
    兼容以下情况：
    - 文本在前，数字标签在后： Text \t 3
    - 标签在前，文本在后：     教育 \t 一段文本...
    - 标签为中文或者数字均可
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    texts, labels = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_samples and len(lines) > max_samples:
        lines = random.sample(lines, max_samples)

    success = 0
    for line in lines:
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

        # A: 标签在后且是数字
        if p2.isdigit():
            label_id = int(p2)
            content = p1

        # B: 标签在前且是中文
        elif p1 in Config.label2id:
            label_id = Config.label2id[p1]
            content = p2

        # C: 标签在前且是数字
        elif p1.isdigit():
            label_id = int(p1)
            content = p2

        # D: 标签在后且是中文
        elif p2 in Config.label2id:
            label_id = Config.label2id[p2]
            content = p1

        # 保存
        if 0 <= label_id < Config.num_classes and content:
            texts.append(content)
            labels.append(label_id)
            success += 1

    if success == 0:
        raise ValueError("未读取到任何有效数据，请检查数据格式或 class_list 是否一致。")

    print(f"[OK] {file_path} 读取成功: {success} 条")
    return texts, labels


# =========================
# 3. 中文清洗
# =========================
def basic_clean(text: str) -> str:
    # 只做轻量清洗，避免过度破坏信息
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# 4. 评估与可视化
# =========================
def evaluate_and_print(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=Config.class_list,
        digits=4
    ))
    return acc, macro_f1


def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path="cm.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # 在格子里写数字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=200)
    print(f"[OK] 混淆矩阵已保存: {save_path}")
    plt.show()


# =========================
# 5. 主流程：TF-IDF + 多种分类器
# =========================
def main():
    print("Loading datasets...")
    train_texts, train_labels = load_data(Config.train_path)
    dev_texts, dev_labels     = load_data(Config.dev_path)
    test_texts, test_labels   = load_data(Config.test_path, max_samples=10000)

    # 清洗
    train_texts = [basic_clean(x) for x in train_texts]
    dev_texts   = [basic_clean(x) for x in dev_texts]
    test_texts  = [basic_clean(x) for x in test_texts]

    # TF-IDF 特征：中文任务常用 char ngram（非常强的传统baseline）
    # 说明：不依赖分词，效果往往比 word-level 更稳
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        min_df=2,
        max_features=200000
    )

    print("Fitting TF-IDF...")
    X_train = vectorizer.fit_transform(train_texts)
    X_dev   = vectorizer.transform(dev_texts)
    X_test  = vectorizer.transform(test_texts)

    # -------- Baseline 1: Linear SVM --------
    svm_clf = LinearSVC(C=1.0)
    print("\nTraining LinearSVC...")
    svm_clf.fit(X_train, train_labels)

    dev_pred = svm_clf.predict(X_dev)
    test_pred = svm_clf.predict(X_test)

    evaluate_and_print("TF-IDF (char 2-5) + LinearSVC | Dev", dev_labels, dev_pred)
    evaluate_and_print("TF-IDF (char 2-5) + LinearSVC | Test", test_labels, test_pred)

    cm = confusion_matrix(test_labels, test_pred)
    plot_confusion_matrix(cm, Config.class_list,
                          title="TF-IDF + LinearSVC (Test)",
                          save_path="cm_svm.png")

    # -------- Baseline 2: Logistic Regression --------
    # 说明：大特征空间建议用 saga，且调高 max_iter
    lr_clf = LogisticRegression(
        C=4.0,
        max_iter=2000,
        n_jobs=-1,
        solver="saga",
        multi_class="auto"
    )
    print("\nTraining LogisticRegression...")
    lr_clf.fit(X_train, train_labels)

    dev_pred = lr_clf.predict(X_dev)
    test_pred = lr_clf.predict(X_test)

    evaluate_and_print("TF-IDF (char 2-5) + LR | Dev", dev_labels, dev_pred)
    evaluate_and_print("TF-IDF (char 2-5) + LR | Test", test_labels, test_pred)

    # -------- Baseline 3: Multinomial Naive Bayes --------
    # 说明：NB 很快，但通常略弱；可作为最低基线
    nb_clf = MultinomialNB(alpha=0.1)
    print("\nTraining MultinomialNB...")
    nb_clf.fit(X_train, train_labels)

    dev_pred = nb_clf.predict(X_dev)
    test_pred = nb_clf.predict(X_test)

    evaluate_and_print("TF-IDF (char 2-5) + MultinomialNB | Dev", dev_labels, dev_pred)
    evaluate_and_print("TF-IDF (char 2-5) + MultinomialNB | Test", test_labels, test_pred)


if __name__ == "__main__":
    main()
