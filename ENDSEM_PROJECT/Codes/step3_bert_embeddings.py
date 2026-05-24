"""
STEP 3 — Extract DNABERT-2 embeddings  (FIXED version)
WHERE TO RUN: Google Colab (free T4 GPU)

INSTALL (run this cell first in Colab):
  !pip install transformers torch numpy

WHY THE FIX: newer transformers versions raise a ValueError when AutoModel
tries to register DNABERT-2's custom BertConfig against the built-in one.
Solution: load BertModel directly — same weights, no conflict.

Output: embeddings_dnabert2.npy  (729 x 768)
"""

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel, BertConfig

# ── Config ────────────────────────────────────────────────────────────────────
FASTA_FILE  = 'hpylori_nonredundant_95.fasta'
OUTPUT_FILE = 'embeddings_dnabert2.npy'
BATCH_SIZE  = 8      # reduce to 4 if CUDA out-of-memory
MAX_LENGTH  = 512

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Load model (FIXED: BertModel instead of AutoModel) ───────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    'zhihan1996/DNABERT-2-117M', trust_remote_code=True
)

print("Loading model...")
config = BertConfig.from_pretrained(
    'zhihan1996/DNABERT-2-117M', trust_remote_code=True
)
model = BertModel.from_pretrained(
    'zhihan1996/DNABERT-2-117M',
    config=config,
    trust_remote_code=True,
    ignore_mismatched_sizes=True
)
model = model.to(device)
model.eval()
print("Model loaded.\n")

# ── FASTA parser ──────────────────────────────────────────────────────────────
def parse_fasta(filepath):
    records = []
    header, seq = '', ''
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    records.append((header, seq))
                header, seq = line, ''
            else:
                seq += line
    if seq:
        records.append((header, seq))
    return records

# ── Embedding extraction ──────────────────────────────────────────────────────
def get_embeddings(sequences):
    """Returns (N, 768) — [CLS] token from last hidden layer."""
    inputs = tokenizer(
        sequences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (batch, seq_len, 768) — position 0 = [CLS] summary
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# ── Main ──────────────────────────────────────────────────────────────────────
records = parse_fasta(FASTA_FILE)
print(f"Loaded {len(records)} sequences")

all_embeddings = []
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    seqs  = [seq.upper() for _, seq in batch]
    emb   = get_embeddings(seqs)
    all_embeddings.append(emb)
    done  = min(i + BATCH_SIZE, len(records))
    print(f"  Embedded {done}/{len(records)}")

embeddings = np.vstack(all_embeddings)
print(f"\nEmbedding matrix shape: {embeddings.shape}")  # (729, 768)

np.save(OUTPUT_FILE, embeddings)
print(f"Saved: {OUTPUT_FILE}")
print("Download this file, put it in your project folder, then run step4_combine_and_train.py")