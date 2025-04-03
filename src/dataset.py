import torch
from torch.utils.data import Dataset
import csv
import random
from random import shuffle
import numpy as np

class PhishingEmailDataset(Dataset):
    def __init__(self, csv_file, vocab=None, max_seq_length=500, augment=False):
        self.max_seq_length = max_seq_length
        self.vocab = vocab
        self.texts = []
        self.labels = []
        self.augment = augment
        self.synonyms = {
            'click': ['tap', 'select', 'press'],
            'link': ['url', 'website', 'webpage'],
            'account': ['profile', 'login', 'credentials'],
            'password': ['passcode', 'pin', 'security code'],
            'verify': ['confirm', 'authenticate', 'validate']
        }

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                text = row[0].lower()
                label = int(row[1])
                self.texts.append(text)
                self.labels.append(label)

        if vocab is None:
            self.build_vocab()

    def build_vocab(self):
        from torchtext.vocab import build_vocab_from_iterator
        counter = build_vocab_from_iterator(
            [text.split() for text in self.texts],
            specials=["<unk>", "<pad>"]
        )
        counter.set_default_index(0)
        self.vocab = counter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.augment and random.random() > 0.5:
            text = self.augment_text(text)

        tokens = text.split()
        tokens = tokens[:self.max_seq_length]
        tokens = [self.vocab[token] if token in self.vocab else self.vocab["<unk>"] for token in tokens]

        if len(tokens) < self.max_seq_length:
            tokens += [self.vocab["<pad>"]] * (self.max_seq_length - len(tokens))

        return torch.tensor(tokens), torch.tensor(label)

    def augment_text(self, text):
        words = text.split()
        
        # 同义词替换
        for i in range(len(words)):
            if words[i] in self.synonyms and random.random() > 0.7:
                words[i] = random.choice(self.synonyms[words[i]])
        
        # 随机插入
        if len(words) < self.max_seq_length and random.random() > 0.5:
            random_word = random.choice(list(self.synonyms.keys()))
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random_word)
        
        # 随机交换
        if len(words) >= 2 and random.random() > 0.8:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)