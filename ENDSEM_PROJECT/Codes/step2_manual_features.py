"""
STEP 2 — Extract manual features (k-mers + structural)
Run: python step2_manual_features.py
Requirements: pip install numpy scikit-learn
Output: features_manual.npy  (729 × 84)
        feature_names.txt
"""

import itertools
import numpy as np
from collections import Counter

# ── Canonical lengths from YOUR dataset ──────────────────────────────────────
CANONICAL_LENGTHS = {
    'babA': 2232, 'cagA': 3540, 'hopA': 1454, 'hopE': 822,
    'hopF': 1443, 'hopG': 1417, 'hopI': 2094, 'hopJ': 1107,
    'hopL': 3693, 'hopM': 2082, 'hopQ': 1924, 'hopZ': 2009,
    'vacA': 3882
}
STOP_CODONS = {'TAA', 'TAG', 'TGA'}
GENES_ORDERED = sorted(CANONICAL_LENGTHS.keys())   # 13 genes → 13 one-hot cols


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP A: Structural / ORF features  (20 features total)
# Why each feature helps:
#   has_start       → complete gene must begin with ATG (or alt. start)
#   has_stop        → complete gene must end with TAA/TAG/TGA
#   internal_stops  → >0 means frameshift or assembly break → fragmented
#   n_count         → any N = gap in assembly → fragmented
#   length_ratio    → <0.8 relative to canonical → likely incomplete
#   gc_content      → H.pylori ~39% GC; outliers suggest contamination
#   seq_length      → raw length; CNN/LSTM use this as a scale signal
#   gene_onehot×13  → each gene has different expected properties
# ─────────────────────────────────────────────────────────────────────────────
def structural_features(seq, gene):
    seq = seq.upper()
    has_start    = int(seq[:3] in ('ATG', 'TTG', 'GTG'))
    has_stop     = int(seq[-3:] in STOP_CODONS)
    internal_stops = sum(1 for i in range(3, len(seq)-3, 3)
                         if seq[i:i+3] in STOP_CODONS)
    n_count      = seq.count('N')
    canonical    = CANONICAL_LENGTHS.get(gene, len(seq))
    length_ratio = round(len(seq) / canonical, 4)
    gc_content   = round((seq.count('G') + seq.count('C')) / max(len(seq), 1), 4)
    seq_length   = len(seq)
    gene_onehot  = [int(g == gene) for g in GENES_ORDERED]

    return [has_start, has_stop, internal_stops, n_count,
            length_ratio, gc_content, seq_length] + gene_onehot


def structural_feature_names():
    base = ['has_start', 'has_stop', 'internal_stops', 'n_count',
            'length_ratio', 'gc_content', 'seq_length']
    onehot = [f'gene_{g}' for g in GENES_ORDERED]
    return base + onehot   # 7 + 13 = 20


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP B: 3-mer frequency features  (64 features)
# Why it helps:
#   Tri-nucleotide frequencies encode codon usage bias.
#   H. pylori virulence genes have characteristic codon usage patterns.
#   A misassembled or chimeric sequence will have disrupted 3-mer profiles.
#   CNN layers learn directly from these local pattern distributions.
# ─────────────────────────────────────────────────────────────────────────────
ALL_3MERS = [''.join(p) for p in itertools.product('ACGT', repeat=3)]  # 64


def kmer_features(seq, k=3):
    seq = seq.upper().replace('N', '')   # ignore ambiguous bases in frequency
    total = max(len(seq) - k + 1, 1)
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    return [round(counts.get(km, 0) / total, 6) for km in ALL_3MERS]


def kmer_feature_names(k=3):
    return [f'kmer_{km}' for km in ALL_3MERS]


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parser
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    fasta_file = 'hpylori_nonredundant_95.fasta'

    records = parse_fasta(fasta_file)
    print(f"Loaded {len(records)} sequences")

    feature_matrix = []
    headers = []

    for i, (header, seq) in enumerate(records):
        gene = header.split('_')[-1]
        struct = structural_features(seq, gene)
        kmers  = kmer_features(seq)
        feature_matrix.append(struct + kmers)
        headers.append(header)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    X = np.array(feature_matrix, dtype=np.float32)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  Structural features : {len(structural_features('ATG', 'cagA'))}")
    print(f"  k-mer features      : {len(kmer_features('ATG'))}")
    print(f"  Total               : {X.shape[1]}")

    # Save feature matrix
    np.save('features_manual.npy', X)
    print("\nSaved: features_manual.npy")

    # Save feature names for interpretability
    names = structural_feature_names() + kmer_feature_names()
    with open('feature_names.txt', 'w') as f:
        for n in names:
            f.write(n + '\n')
    print("Saved: feature_names.txt")

    # Save header order (so rows align with labels.csv)
    with open('sequence_order.txt', 'w') as f:
        for h in headers:
            f.write(h + '\n')
    print("Saved: sequence_order.txt")
    print("\nDone! Run step3_bert_embeddings.py next (needs GPU/Colab)")


if __name__ == '__main__':
    main()
