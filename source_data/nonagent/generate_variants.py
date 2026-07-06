"""Generate single-position mutation variants for all selected peptides."""
import csv
import json
from pathlib import Path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "selected_peptides_for_optimizer_comparison.csv"
OUT_DIR = Path(__file__).resolve().parent / "work"
OUT_DIR.mkdir(exist_ok=True)


def mutate_one_position(seq):
    variants = []
    for i, old in enumerate(seq):
        for aa in AMINO_ACIDS:
            if aa != old:
                variants.append(seq[:i] + aa + seq[i+1:])
    return variants


def main():
    rows = []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    all_variants = set()
    per_peptide = []
    for row in rows:
        seq = row["sequence"]
        variants = mutate_one_position(seq)
        per_peptide.append({
            "start_id": row["start_id"],
            "sequence": seq,
            "variants": variants,
        })
        all_variants.update(variants)
        all_variants.add(seq)

    all_variants = sorted(all_variants)
    print(f"{len(rows)} starting peptides")
    print(f"{sum(len(p['variants']) for p in per_peptide)} variants total (with duplicates)")
    print(f"{len(all_variants)} unique sequences to predict (incl. originals)")

    (OUT_DIR / "per_peptide.json").write_text(json.dumps(per_peptide, indent=2))
    (OUT_DIR / "unique_sequences.json").write_text(json.dumps({"sequences": all_variants}))

    # Also write a comma-joined version for convenient MCP batch input.
    (OUT_DIR / "unique_sequences.txt").write_text(",".join(all_variants))


if __name__ == "__main__":
    main()
