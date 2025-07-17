#!/usr/bin/env python3
# Created by Sean L. on Jul. 16.
# Last Updated by Sean L. on Jul. 16.
# 
# TeaML
# embedding/train.py
# 
# Makabaka1880, 2025. All rights reserved.

from typing import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import *;

# MARK: ─── Training Logic ─────────────────────────────────────────────────────
def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for tokens, params in dataloader:
        input_seq = tokens[:, :-1].to(device)
        param_seq = params[:, :-1].to(device)
        target_seq = tokens[:, 1:].to(device)

        optimizer.zero_grad()
        hidden = encoder(input_seq, param_seq)
        logits, _ = decoder(input_seq, hidden)

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * tokens.size(0)

    return total_loss

# MARK: ─── Entrypoint ────────────────────────────────────────────────────────
def main():
    dataset = ProcDataset('main.db', vocab_to_idx, START_IDX, END_IDX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    encoder = ProcEncoder(len(vocabs), EMBED_DIM, HIDDEN_DIM).to(device)
    decoder = ProcDecoder(len(vocabs), EMBED_DIM, HIDDEN_DIM, device, encoder.token_embed).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print("Starting training...")
    loss_array = []
    for epoch in range(10):
        epoch_loss = train_one_epoch(encoder, decoder, dataloader, optimizer, loss_fn)
        loss_array.append(epoch_loss)
        print(f"[{epoch:04d}] Loss = {epoch_loss:.4f}")
    
    torch.save(encoder.state_dict(), "model_weights.pt")

if __name__ == "__main__":
    main()