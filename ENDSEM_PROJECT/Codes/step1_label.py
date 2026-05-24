"""
STEP 1 — Auto-label all 729 sequences
Run: python step1_label.py
Output: labels.csv
"""

import csv

# Canonical (median) lengths measured from YOUR actual dataset
CANONICAL_LENGTHS = {
    'babA': 2232, 'cagA': 3540, 'hopA': 1454, 'hopE': 822,
    'hopF': 1443, 'hopG': 1417, 'hopI': 2094, 'hopJ': 1107,
    'hopL': 3693, 'hopM': 2082, 'hopQ': 1924, 'hopZ': 2009,
    'vacA': 3882
}
STOP_CODONS = {'TAA', 'TAG', 'TGA'}


def assign_label(seq, gene):
    """
    Rules (biologically grounded):
      fragmented  → has in-frame internal stop codons OR ambiguous N bases
      incomplete  → missing start/stop codon OR length < 80% of canonical
      complete    → passes all checks
    """
    seq = seq.upper()
    has_start = seq[:3] in ('ATG', 'TTG', 'GTG')
    has_stop  = seq[-3:] in STOP_CODONS
    n_count   = seq.count('N')
    internal_stops = sum(
        1 for i in range(3, len(seq) - 3, 3)
        if seq[i:i+3] in STOP_CODONS
    )
    canonical = CANONICAL_LENGTHS.get(gene, len(seq))
    length_ratio = len(seq) / canonical

    if internal_stops > 0 or n_count > 0:
        return 'fragmented'
    if not has_start or not has_stop or length_ratio < 0.80:
        return 'incomplete'
    return 'complete'


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


def main():
    fasta_file = 'hpylori_nonredundant_95.fasta'
    output_csv = 'labels.csv'

    records = parse_fasta(fasta_file)
    print(f"Loaded {len(records)} sequences")

    from collections import Counter
    label_counts = Counter()
    rows = []

    for header, seq in records:
        gene = header.split('_')[-1]
        label = assign_label(seq, gene)
        label_counts[label] += 1
        rows.append({
            'header':       header,
            'gene':         gene,
            'length':       len(seq),
            'canonical':    CANONICAL_LENGTHS.get(gene, 'unknown'),
            'length_ratio': round(len(seq) / CANONICAL_LENGTHS.get(gene, len(seq)), 3),
            'has_start':    int(seq.upper()[:3] in ('ATG','TTG','GTG')),
            'has_stop':     int(seq.upper()[-3:] in STOP_CODONS),
            'internal_stops': sum(1 for i in range(3,len(seq)-3,3) if seq.upper()[i:i+3] in STOP_CODONS),
            'n_count':      seq.upper().count('N'),
            'label':        label
        })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLabel distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl:12s}: {cnt:3d}  ({cnt/len(records)*100:.1f}%)")
    print(f"\nSaved to {output_csv}")
    print("Next: manually verify ~30 edge cases (especially 'fragmented' ones)")


if __name__ == '__main__':
    main()
