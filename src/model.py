import torch
import torch.nn as nn

class PhishingEmailClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(PhishingEmailClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, _) = self.lstm(embedded)
        lstm_out = self.dropout1(lstm_out)
        output = self.fc(hidden.squeeze(0))
        output = self.dropout2(output)
        output = self.sigmoid(output)
        return output