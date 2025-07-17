# 🍵 Tea Grading

A pytorch model for grading tea quality based on a sequence of procedures.

## 📖 Project Structure

```
- tea_grading/
    - load_training.py
    - train.py
    - main.db
    - schemas.sqlite3-query
    - tokens.json
```
- **load_training.py**: Script to load training data.
- **train.py**: Script to train the model.
- **main.db**: SQLite database containing the training data.
- **schemas.sqlite3-query**: SQL queries for the database schema.
- **tokens.json**: JSON file containing tokens for the model.

## 🏠 Architecture

The model is a **sequence autoencoder** with three parts: tokenizer, encoder, and decoder.

### 🔧 Tokenizer

- Normalizes step parameters (`param1`, `param2`, `param3`) using global min-max scaling.
- Converts normalized parameters into fixed-length decimal-string tokens.
- Combines each operation token with its parameter tokens into a flat token sequence.
- Wraps the sequence with special `<START>` and `<END>` tokens.

### 💻 Encoder

- Embeds tokens and projects parameter triplets separately.
- Adds token and parameter embeddings element-wise.
- Feeds combined embeddings into a GRU to produce a latent representation.

### 🔒 Decoder

- Takes the token sequence (excluding last token) and latent vector as input.
- Uses the shared token embedding and a GRU to generate output sequences.
- Applies a linear layer to produce logits for next-token prediction.

## 🔮 Output
The semantic output is in the hidden state of the encoder(`ProcEncoder`), which captures the sequence of operations and their parameters. The three components contains
1. Phenol / 500g (mg)
2. Caffeine / 500g (mg)
3. Price / 500g (RMB)

## ✍️ License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.