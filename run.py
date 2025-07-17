# Created by Sean L. on Jul. 18.
# Last Updated by Sean L. on Jul. 18.
# 
# TeaML
# run.py
# 
# Makabaka1880, 2025. All rights reserved.

from models import *;
import torch
import sys

try:
    encoder = ProcEncoder(len(vocabs), EMBED_DIM, HIDDEN_DIM).to(device)
    encoder.load_state_dict(torch.load('model_weights.pt', map_location=device, weights_only=False))
    encoder.eval()
except Exception as e:
    print(json.dumps({"error": "model load error", "detail": str(e)}))
    exit(1)

token_ids = []
param_list = []

try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        token, p1, p2, p3 = line.split(',')
        token_id = vocab_to_idx.get(token.strip(), vocab_to_idx["<PAD>"])
        token_ids.append(token_id)
        param_list.append([float(p1), float(p2), float(p3)])

except Exception as e:
    print(json.dumps({"error": "input parse error", "detail": str(e)}))
    exit(1)

if not token_ids:
    print(json.dumps({"error": "no valid input"}))
    exit(1)

try:
    token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    param_tensor = torch.tensor([param_list], dtype=torch.float32, device=device)

    with torch.no_grad():
        hidden = encoder.get_embedded(encoder(token_tensor, param_tensor))

    # Assume hidden is (1, H) or (num_layers, batch, hidden_size)
    if isinstance(hidden, tuple):  # e.g., GRU returns (output, hidden)
        hidden = hidden[1]

    output_list = hidden.squeeze(0).cpu().tolist()
    print(json.dumps({"embedding": output_list}))
except Exception as e:
    print(json.dumps({"error": "forward pass error", "detail": str(e)}))
    exit(1)

exit(0)