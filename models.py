# Created by Sean L. on Jul. 18.
# Last Updated by Sean L. on Jul. 18.
# 
# TeaML
# models.py
# 
# Makabaka1880, 2025. All rights reserved.

import sqlite3
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from torch.nn.utils.rnn import pad_sequence

# MARK: ─── Hyperparameters & Device ──────────────────────────────────────────
EMBED_DIM = 128
HIDDEN_DIM = 16
BATCH_SIZE = 2

device = torch.device("mps" if torch.backends.mps.is_available() and BATCH_SIZE > 8 else "cpu")

# MARK: ─── Vocabulary ────────────────────────────────────────────────────────
with open("tokens.json") as tokenfile:
    vocab_list = json.load(tokenfile)

vocabs = dict(enumerate(vocab_list))
vocab_to_idx = {v: k for k, v in vocabs.items()}

START_IDX = vocab_to_idx["<START>"]
END_IDX   = vocab_to_idx["<END>"]
PAD_IDX   = vocab_to_idx["<PAD>"]

def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = [item["tokens"] for item in batch]
    params = [item["params"] for item in batch]

    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=PAD_IDX)
    padded_params = pad_sequence(params, batch_first=True, padding_value=0.0)

    return padded_tokens, padded_params

# MARK: ─── Dataset ───────────────────────────────────────────────────────────
class ProcDataset(Dataset):
    def __init__(self, db_path: str, vocab_to_idx: dict, start_idx: int, end_idx: int):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.vocab_to_idx = vocab_to_idx
        self.START_IDX = start_idx
        self.END_IDX = end_idx

        # Fetch all procedure IDs
        self.cursor.execute("SELECT id FROM procedures")
        self.procedure_ids = [row[0] for row in self.cursor.fetchall()]

        # Compute global min and max for each param
        self.cursor.execute("""
            SELECT 
                MIN(param1), MAX(param1), 
                MIN(param2), MAX(param2), 
                MIN(param3), MAX(param3)
            FROM steps
        """)
        (self.min1, self.max1,
            self.min2, self.max2,
            self.min3, self.max3) = self.cursor.fetchone()

    def normalize(self, val: float, min_val: float, max_val: float) -> float:
        return (val - min_val) / (max_val - min_val + 1e-8)

    def __len__(self):
        return len(self.procedure_ids)

    def __getitem__(self, idx: int):
        proc_id = self.procedure_ids[idx]

        self.cursor.execute("""
            SELECT operation, param1, param2, param3
            FROM steps
            WHERE procedure_id = ?
            ORDER BY step_idx ASC
        """, (proc_id,))
        steps = self.cursor.fetchall()

        token_ids = [self.START_IDX]
        param_tuples = [torch.tensor([0.0, 0.0, 0.0])]  # <START> token has no params

        for op, p1, p2, p3 in steps:
            token_ids.append(self.vocab_to_idx.get(op, self.vocab_to_idx["<PAD>"]))

            if p1 != 0.0 or p2 != 0.0 or p3 != 0.0:
                norm_p1 = self.normalize(p1, self.min1, self.max1)
                norm_p2 = self.normalize(p2, self.min2, self.max2)
                norm_p3 = self.normalize(p3, self.min3, self.max3)
                param_tuples.append(torch.tensor([norm_p1, norm_p2, norm_p3]))
            else:
                param_tuples.append(torch.tensor([0.0, 0.0, 0.0]))

        token_ids.append(self.END_IDX)
        param_tuples.append(torch.tensor([0.0, 0.0, 0.0]))  # <END> token has no params

        return {
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "params": torch.stack(param_tuples)
        }

    def __del__(self):
        self.conn.close()


# MARK: ─── Models ────────────────────────────────────────────────────────────
class ProcEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, proj_dim=3):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.param_embed = nn.Linear(3, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, token_ids, param_tensor):  # param_tensor shape: [B, L, 3]
        token_emb = self.token_embed(token_ids)
        param_emb = self.param_embed(param_tensor)
        emb = token_emb + param_emb  # or concat instead
        _, hidden = self.gru(emb)
        return hidden
    
    def get_embedded(self, hidden):
        hidden = hidden.unsqueeze(0)
        return self.proj(hidden)

class ProcDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, device, embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.embedding = embedding or nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.device = device

    def forward(self, input_tokens: torch.Tensor, hidden: torch.Tensor):
        emb = self.embedding(input_tokens.to(self.device))
        out, hidden = self.decoder(emb, hidden.to(self.device))
        logits = self.output_proj(out)
        return logits, hidden