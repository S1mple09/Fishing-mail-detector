import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import copy

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in dataloader:
        text, label = batch
        text = text.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label.float())

        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == label).float()
        acc = correct.sum() / len(correct)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            text, label = batch
            text = text.to(device)
            label = label.to(device)

            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label.float())

            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == label).float()
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), precision, recall, f1

def k_fold_cross_validation(model, dataset, n_splits=5, epochs=10, lr=0.001, weight_decay=0.01):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    results = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=test_subsampler)
        
        model_instance = copy.deepcopy(model)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopper = EarlyStopper(patience=3, min_delta=0.001)
        
        for epoch in range(epochs):
            train_loss, train_acc = train(model_instance, train_loader, optimizer, criterion, 'cpu')
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model_instance, test_loader, criterion, 'cpu')
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if early_stopper.early_stop(val_loss):
                print("Early stopping triggered")
                break
        
        results.append({
            'fold': fold+1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
    
    return results