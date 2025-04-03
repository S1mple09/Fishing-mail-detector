import torch
import os

def save_model(model, save_path):
    """保存模型权重"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path, device):
    """加载模型权重"""
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {load_path}")
    return model