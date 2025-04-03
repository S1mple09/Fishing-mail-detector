import torch
import pandas as pd
from src.dataset import PhishingEmailDataset
from src.model import PhishingEmailClassifier
from src.utils import load_model

def predict_new_emails(model_path, input_csv, vocab, max_seq_length=5000):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(vocab)
    model = PhishingEmailClassifier(vocab_size, embedding_dim=128, hidden_dim=64, output_dim=1).to(device)
    load_model(model, model_path, device)

    # 加载输入数据
    df = pd.read_csv(input_csv)
    texts = df['text'].tolist()

    # 预处理输入数据
    processed_texts = []
    for text in texts:
        tokens = text.lower().split()
        tokens = tokens[:max_seq_length]  # 截断过长的文本
        tokens = [vocab[token] if token in vocab else vocab["<unk>"] for token in tokens]
        if len(tokens) < max_seq_length:
            tokens += [vocab["<pad>"]] * (max_seq_length - len(tokens))
        processed_texts.append(torch.tensor(tokens))

    # 预测
    model.eval()
    with torch.no_grad():
        for i, text_tensor in enumerate(processed_texts):
            text_tensor = text_tensor.unsqueeze(0).to(device)  # 添加批次维度
            prediction = model(text_tensor)
            predicted_label = 1 if prediction.item() > 0.5 else 0
            print(f"邮件: {texts[i]}")
            print(f"预测结果: {'钓鱼邮件' if predicted_label == 1 else '正常邮件'}\n")

# 示例调用
if __name__ == "__main__":
    # 假设已经有一个训练好的词汇表
    train_dataset = PhishingEmailDataset("data/train_data.csv", max_seq_length=5000)
    vocab = train_dataset.vocab

    # 预测新的邮件
    predict_new_emails(
        model_path="models/phishing_email_model.pth",
        input_csv="data/input.csv",
        vocab=vocab
    )