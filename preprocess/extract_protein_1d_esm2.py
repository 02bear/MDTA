from pathlib import Path
import json
import pandas as pd
import torch
import esm
from tqdm import tqdm


def load_esm2(model_name="esm2_t33_650M_UR50D"):
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        repr_layer = 33
    elif model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        repr_layer = 36
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, alphabet, batch_converter, device, repr_layer


def safe_filename(name: str):
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def extract_one_protein(model, alphabet, batch_converter, device, repr_layer, protein_id, sequence, max_length=1200):
    original_seq = str(sequence).strip()
    used_seq = original_seq[:max_length]
    truncated = len(original_seq) > max_length

    batch_labels, batch_strs, batch_tokens = batch_converter([(protein_id, used_seq)])
    batch_tokens = batch_tokens.to(device)

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

    token_representations = results["representations"][repr_layer]  # [1, T, D]
    tokens_len = int(batch_lens[0].item())

    # 官方说明：token 0 是 BOS，所以真实残基是 [1 : tokens_len - 1]
    per_tok = token_representations[0, 1:tokens_len - 1].detach().cpu()  # [L, D]
    mean_repr = per_tok.mean(dim=0)  # [D]

    return {
        "protein_id": protein_id,
        "orig_len": len(original_seq),
        "used_len": len(used_seq),
        "truncated": truncated,
        "per_tok": per_tok,
        "mean": mean_repr,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--id_col", type=str, default="protein_id")
    parser.add_argument("--seq_col", type=str, default="sequence")
    parser.add_argument("--max_length", type=int, default=1200)
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D",
        choices=["esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"],
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[[args.id_col, args.seq_col]].drop_duplicates().reset_index(drop=True)

    model, alphabet, batch_converter, device, repr_layer = load_esm2(args.model_name)

    failures = []
    success_count = 0
    truncated_count = 0
    embed_dim = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ESM-2 protein embeddings"):
        protein_id = str(row[args.id_col])
        sequence = str(row[args.seq_col])

        try:
            out = extract_one_protein(
                model=model,
                alphabet=alphabet,
                batch_converter=batch_converter,
                device=device,
                repr_layer=repr_layer,
                protein_id=protein_id,
                sequence=sequence,
                max_length=args.max_length,
            )

            save_path = output_dir / f"{safe_filename(protein_id)}.pt"
            torch.save(out, save_path)

            success_count += 1
            if out["truncated"]:
                truncated_count += 1
            if embed_dim is None:
                embed_dim = int(out["per_tok"].shape[1])

        except Exception as e:
            failures.append({
                "protein_id": protein_id,
                "error": str(e),
            })

    fail_path = output_dir / "failures.csv"
    pd.DataFrame(failures).to_csv(fail_path, index=False)

    meta = {
        "model_name": args.model_name,
        "repr_layer": repr_layer,
        "num_input_proteins": int(len(df)),
        "num_success": int(success_count),
        "num_failed": int(len(failures)),
        "num_truncated": int(truncated_count),
        "embedding_dim": embed_dim,
        "max_length": int(args.max_length),
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved embeddings to: {output_dir}")
    print(f"Failure log: {fail_path}")


if __name__ == "__main__":
    main()