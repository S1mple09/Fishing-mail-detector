import torch
from torch.utils.data import DataLoader
from src.dataset import PhishingEmailDataset
from src.model import PhishingEmailClassifier
from src.trainer import train, evaluate
from src.utils import save_model
import os

# 超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
MAX_SEQ_LENGTH = 5000  # 邮件文本的最大长度
MODEL_SAVE_PATH = "models/phishing_email_model.pth"


def main():
    # 数据路径
    train_csv = "data/train_data.csv"
    test_csv = "data/test_data.csv"

    # 加载数据集
    train_dataset = PhishingEmailDataset(train_csv, max_seq_length=MAX_SEQ_LENGTH)
    test_dataset = PhishingEmailDataset(test_csv, vocab=train_dataset.vocab, max_seq_length=MAX_SEQ_LENGTH)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型参数
    vocab_size = len(train_dataset.vocab)
    output_dim = 1  # 二分类问题

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhishingEmailClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    # 训练模型
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, precision, recall, f1 = evaluate(model, test_loader, criterion, device)

        print(f'第 {epoch + 1:02} 轮训练')
        print(f'\t训练损失: {train_loss:.3f} | 训练准确率: {train_acc * 100:.2f}%')
        print(f'\t测试损失: {test_loss:.3f} | 测试准确率: {test_acc * 100:.2f}%')
        print(f'\t精确率: {precision:.3f} | 召回率: {recall:.3f} | F1 分数: {f1:.3f}')

    # 保存模型
    save_model(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()