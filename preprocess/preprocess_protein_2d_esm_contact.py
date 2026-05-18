import os
import pickle
from pathlib import Path

import esm
import pandas as pd
import torch
from tqdm import tqdm


def load_esm_model(model_name="esm2_t33_650M_UR50D"):
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, alphabet, batch_converter, device


def get_esm_contact_map(
    model,
    alphabet,
    batch_converter,
    device,
    input_csv,
    output_pkl,
    id_col="protein_id",
    seq_col="sequence",
    max_length=1200,
):
    df = pd.read_csv(input_csv)
    prot_data = []

    for prot_id, seq in zip(df[id_col], df[seq_col]):
        seq = str(seq)
        truncated_seq = seq[:max_length]
        prot_data.append((str(prot_id), truncated_seq))

    target_graph = {}
    length_target = {}

    for prot_id, seq in tqdm(prot_data, desc="Generating ESM contact maps"):
        batch_labels, batch_strs, batch_tokens = batch_converter([(prot_id, seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, return_contacts=True)

        contact_map = results["contacts"][0].detach().cpu().numpy()

        target_graph[prot_id] = contact_map
        length_target[prot_id] = len(seq)

    dump_data = {
        "contact_map": target_graph,
        "length_dict": length_target,
        "model_name": type(model).__name__,
        "max_length": max_length,
        "input_csv": str(input_csv),
    }

    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(dump_data, f)

    print(f"Saved contact maps to: {output_pkl}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_pkl", type=str, required=True)
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

    model, alphabet, batch_converter, device = load_esm_model(args.model_name)

    get_esm_contact_map(
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
        device=device,
        input_csv=args.input_csv,
        output_pkl=args.output_pkl,
        id_col=args.id_col,
        seq_col=args.seq_col,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()