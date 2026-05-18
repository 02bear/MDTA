# -*- coding: utf-8 -*-
import re
import argparse
import pandas as pd


def to_base_id(pid: str):
    pid = str(pid).strip()
    # 去掉括号里的突变标记，如 ABL1(E255K) -> ABL1
    pid = re.sub(r"\(.*?\)", "", pid)
    # 去掉末尾的 p，如 ABL1p -> ABL1
    pid = re.sub(r"p$", "", pid)
    return pid.strip()


def main():
    parser = argparse.ArgumentParser(description="Build base protein IDs from proteins.csv")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to proteins.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output stage1 csv")
    parser.add_argument("--id_col", type=str, default="protein_id", help="Protein ID column name")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns:
        raise ValueError(f"Column '{args.id_col}' not found in {args.input_csv}")

    df["base_id"] = df[args.id_col].apply(to_base_id)
    df["is_mutant_like"] = df[args.id_col] != df["base_id"]

    print(df[[args.id_col, "base_id", "is_mutant_like"]].head(20).to_string(index=False))
    print("\nN total:", len(df))
    print("N unique base_id:", df["base_id"].nunique())

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
