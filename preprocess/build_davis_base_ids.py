# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/raw/davis/proteins.csv")

def to_base_id(pid: str):
    pid = str(pid).strip()
    # 去掉括号里的突变标记，如 ABL1(E255K) -> ABL1
    pid = re.sub(r"\(.*?\)", "", pid)
    # 去掉末尾的 p，如 ABL1p -> ABL1
    pid = re.sub(r"p$", "", pid)
    return pid.strip()

df["base_id"] = df["protein_id"].apply(to_base_id)
df["is_mutant_like"] = df["protein_id"] != df["base_id"]

print(df[["protein_id", "base_id", "is_mutant_like"]].head(20).to_string(index=False))
print("\nN total:", len(df))
print("N unique base_id:", df["base_id"].nunique())

df.to_csv("data/raw/davis/protein_mapping_stage1.csv", index=False)
print("\nSaved: data/raw/davis/protein_mapping_stage1.csv")