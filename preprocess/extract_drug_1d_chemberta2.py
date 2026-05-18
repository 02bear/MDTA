from pathlib import Path
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem


def safe_filename(name: str):
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def canonicalize_smiles(smiles: str):
    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mean_pool(last_hidden_state, attention_mask):
    """
    last_hidden_state: [1, T, D]
    attention_mask:    [1, T]
    """
    mask = attention_mask.unsqueeze(-1).float()        # [1, T, 1]
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)                         # [1, D]
    counts = mask.sum(dim=1).clamp(min=1e-9)          # [1, 1]
    return summed / counts                             # [1, D]


def extract_one_smiles(model, tokenizer, device, drug_id, smiles, max_length=256):
    encoded = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    last_hidden = outputs.last_hidden_state.detach().cpu()          # [1, T, D]
    attention_mask = encoded["attention_mask"].detach().cpu()       # [1, T]

    pooled_mean = mean_pool(last_hidden, attention_mask).squeeze(0) # [D]
    cls_repr = last_hidden[0, 0, :]                                 # [D]

    token_len = int(attention_mask.sum().item())

    return {
        "drug_id": drug_id,
        "smiles": smiles,
        "used_token_len": token_len,
        "max_length": max_length,
        "per_tok": last_hidden.squeeze(0),            # [T, D]
        "attention_mask": attention_mask.squeeze(0),  # [T]
        "mean": pooled_mean,                          # [D]
        "cls": cls_repr,                              # [D]
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--id_col", type=str, default="drug_id")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument(
        "--model_name",
        type=str,
        default="seyonec/ChemBERTa_zinc250k_v2_40k",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--canonicalize", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[[args.id_col, args.smiles_col]].drop_duplicates().reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    failures = []
    success_count = 0
    truncated_count = 0
    embedding_dim = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ChemBERTa drug embeddings"):
        drug_id = str(row[args.id_col])
        smiles = str(row[args.smiles_col])

        if args.canonicalize:
            cano = canonicalize_smiles(smiles)
            if cano is None:
                failures.append({
                    "drug_id": drug_id,
                    "smiles": smiles,
                    "error": "RDKit canonicalization failed"
                })
                continue
            smiles_used = cano
        else:
            smiles_used = smiles

        try:
            out = extract_one_smiles(
                model=model,
                tokenizer=tokenizer,
                device=device,
                drug_id=drug_id,
                smiles=smiles_used,
                max_length=args.max_length,
            )

            # 简单判断是否触发截断
            if out["used_token_len"] >= args.max_length:
                truncated_count += 1

            save_path = output_dir / f"{safe_filename(drug_id)}.pt"
            torch.save(out, save_path)

            success_count += 1
            if embedding_dim is None:
                embedding_dim = int(out["mean"].shape[0])

        except Exception as e:
            failures.append({
                "drug_id": drug_id,
                "smiles": smiles_used,
                "error": str(e),
            })

    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    meta = {
        "model_name": args.model_name,
        "num_input_drugs": int(len(df)),
        "num_success": int(success_count),
        "num_failed": int(len(failures)),
        "num_maybe_truncated": int(truncated_count),
        "embedding_dim": embedding_dim,
        "max_length": int(args.max_length),
        "canonicalize": bool(args.canonicalize),
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved embeddings to: {output_dir}")
    print(f"Failure log: {output_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()