# Created by Sean L. on Jul. 14.
# Last Updated by Sean L. on Jul. 14.
# 
# TeaML
# embedding/train.py
# 
# Makabaka1880, 2025. All rights reserved.

from typing import *
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# MARK: Vocab loading
with open("tokens.json") as tokenfile:
    content = str(tokenfile.read())
    vocabs = dict(enumerate(json.loads(content)))
    vocab_to_idx = {v: k for k, v in vocabs.items()}


START_IDX = vocab_to_idx["<START>"]
PAD_IDX = vocab_to_idx["<PAD>"]
END_IDX = vocab_to_idx["<END>"]
EMBED_DIM = 128
HIDDEN_DIM = 16
BATCH_SIZE = 2

device = torch.device("mps" if (torch.backends.mps.is_available() and BATCH_SIZE > 8) else "cpu")
print("Using device:", device)
# MARK: Models
class ProcEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, token_ids: torch.Tensor, device: torch.device):
        token_ids = token_ids.to(device)
        emb = self.embedding(token_ids)
        emb = emb.to(device)
        _, h = self.encoder(emb)
        return h

class ProcDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_tokens: torch.Tensor, hidden: torch.Tensor, device: torch.device):
        input_tokens = input_tokens.to(device)
        hidden = hidden.to(device)
        emb = self.embedding(input_tokens)
        emb = emb.to(device)
        out, hidden = self.decoder(emb, hidden)
        logits = self.output_proj(out)
        return logits, hidden


class ProcDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.dataset = []
        with open(path, 'r') as f:
            self.dataset = json.load(f)
        self.dataset = list(map(
            lambda x: {
                **x,
                "sequence": [START_IDX] + list(map(lambda n: vocab_to_idx[n], x["sequence"])) + [END_IDX]
            },
            self.dataset
        ))


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

def collate_fn(batch):
    sequences = [torch.tensor(item['sequence'], dtype=torch.long) for item in batch]
    padded = pad_sequence(sequences, batch_first=True, padding_value=vocab_to_idx["<PAD>"])
    return padded


def train_one_epoch(
        encoder: nn.Module, 
        decoder: nn.Module, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ):

    encoder.train()
    decoder.train()
    total_loss = 0
    for batch in dataloader:
        batch.to(device)
        input_seq = batch[:, :-1].to(device)
        target_seq = batch[:, 1:].to(device)

        optimizer.zero_grad()

        hidden = encoder(input_seq, device)
        logits, _ = decoder(input_seq, hidden, device)

        logits = logits.reshape(-1, logits.size(-1))
        targets = target_seq.reshape(-1)

        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss


# MARK: Train
dataset = ProcDataset('training.json')
dataloader = DataLoader(dataset, BATCH_SIZE, collate_fn=collate_fn)

encoder = ProcEncoder(len(vocabs), EMBED_DIM, HIDDEN_DIM).to(device)
decoder = ProcDecoder(len(vocabs), EMBED_DIM, HIDDEN_DIM).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

loss_array = []
for i in range(1000):
    loss = train_one_epoch(encoder, decoder, dataloader, optimizer, loss_fn)
    loss_array.append(loss)
    print(loss)