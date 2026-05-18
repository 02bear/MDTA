from pathlib import Path
import pandas as pd


def parse_txt_to_df(txt_path: Path) -> pd.DataFrame:
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 每行按空白切分，前2列固定，后1列是label，中间两列分别是smiles和sequence
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] skip malformed line {line_idx} in {txt_path.name}: {line[:100]}")
                continue

            drug_id = parts[0]
            protein_id = parts[1]
            smiles = parts[2]
            label = parts[-1]
            sequence = "".join(parts[3:-1])  # 理论上通常只有一列，这里保险一点

            try:
                label = float(label)
            except ValueError:
                print(f"[WARN] skip bad label line {line_idx} in {txt_path.name}: {label}")
                continue

            rows.append(
                {
                    "drug_id": drug_id,
                    "protein_id": protein_id,
                    "smiles": smiles,
                    "sequence": sequence,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    return df


def save_dataset(df: pd.DataFrame, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 主表
    pairs_path = out_dir / "pairs.csv"
    df.to_csv(pairs_path, index=False)

    # 去重药物表
    drugs_df = df[["drug_id", "smiles"]].drop_duplicates().reset_index(drop=True)
    drugs_path = out_dir / "drugs.csv"
    drugs_df.to_csv(drugs_path, index=False)

    # 去重蛋白表
    proteins_df = df[["protein_id", "sequence"]].drop_duplicates().reset_index(drop=True)
    proteins_path = out_dir / "proteins.csv"
    proteins_df.to_csv(proteins_path, index=False)

    print(f"\n[{prefix}]")
    print(f"pairs     : {pairs_path}  ({len(df)} rows)")
    print(f"drugs     : {drugs_path}  ({len(drugs_df)} unique drugs)")
    print(f"proteins  : {proteins_path}  ({len(proteins_df)} unique proteins)")


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_root = project_root / "data" / "raw"

    # DAVIS
    davis_txt = raw_root / "davis" / "davis.txt"
    davis_filter_txt = raw_root / "davis" / "davis-filter.txt"

    if davis_txt.exists():
        davis_df = parse_txt_to_df(davis_txt)
        save_dataset(davis_df, raw_root / "davis", "DAVIS full")

    if davis_filter_txt.exists():
        davis_filter_df = parse_txt_to_df(davis_filter_txt)
        save_dataset(davis_filter_df, raw_root / "davis_filter", "DAVIS filtered")

    # KIBA
    kiba_txt = raw_root / "kiba" / "kiba.txt"
    if kiba_txt.exists():
        kiba_df = parse_txt_to_df(kiba_txt)
        save_dataset(kiba_df, raw_root / "kiba", "KIBA")


if __name__ == "__main__":
    main()