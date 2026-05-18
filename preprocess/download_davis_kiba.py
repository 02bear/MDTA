import os
import shutil
from pathlib import Path

import kagglehub


DATASET_NAME = "christang0002/davis-and-kiba"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def write_manifest(root: Path, out_file: Path):
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir():
            lines.append(f"[DIR]  {rel}")
        else:
            lines.append(f"[FILE] {rel}")
    out_file.write_text("\n".join(lines), encoding="utf-8")


def classify_path(path: Path):
    """
    根据路径名判断属于 davis / kiba / unknown
    """
    parts = [x.lower() for x in path.parts]
    joined = "/".join(parts)

    if "davis" in joined:
        return "davis"
    if "kiba" in joined:
        return "kiba"
    return "unknown"


def get_relative_after_keyword(path: Path, keyword: str):
    """
    把路径裁剪成 keyword 后面的相对路径
    例如:
    /xxx/DAVIS/folds/train_fold_setting1.txt
    -> folds/train_fold_setting1.txt
    """
    parts = list(path.parts)
    parts_lower = [x.lower() for x in parts]
    if keyword in parts_lower:
        idx = parts_lower.index(keyword)
        tail = parts[idx + 1 :]
        if len(tail) == 0:
            return Path(path.name)
        return Path(*tail)
    return Path(path.name)


def copy_dataset_files(download_root: Path, project_root: Path):
    raw_root = project_root / "data" / "raw"
    davis_dst = raw_root / "davis"
    kiba_dst = raw_root / "kiba"

    ensure_dir(davis_dst)
    ensure_dir(kiba_dst)

    copied_davis = 0
    copied_kiba = 0
    unknown_files = []

    all_files = [p for p in download_root.rglob("*") if p.is_file()]

    if not all_files:
        raise RuntimeError(f"下载目录里没找到任何文件：{download_root}")

    for f in all_files:
        cls = classify_path(f)

        if cls == "davis":
            rel = get_relative_after_keyword(f, "davis")
            dst = davis_dst / rel
            copy_file(f, dst)
            copied_davis += 1

        elif cls == "kiba":
            rel = get_relative_after_keyword(f, "kiba")
            dst = kiba_dst / rel
            copy_file(f, dst)
            copied_kiba += 1

        else:
            unknown_files.append(f)

    return copied_davis, copied_kiba, unknown_files


def fallback_copy_unknowns(unknown_files, project_root: Path):
    """
    如果有一些文件路径里不带 davis/kiba，
    那就先统一放到 misc 里，避免丢文件。
    """
    misc_root = project_root / "data" / "raw" / "_unclassified"
    ensure_dir(misc_root)

    copied = 0
    for f in unknown_files:
        dst = misc_root / f.name
        base = dst.stem
        suffix = dst.suffix
        i = 1
        while dst.exists():
            dst = misc_root / f"{base}_{i}{suffix}"
            i += 1
        copy_file(f, dst)
        copied += 1
    return copied


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_root = project_root / "data" / "raw"
    ensure_dir(raw_root)

    print(f"[1/4] Downloading dataset from KaggleHub: {DATASET_NAME}")
    download_path = kagglehub.dataset_download(DATASET_NAME)
    download_root = Path(download_path).resolve()
    print(f"Downloaded to: {download_root}")

    print("[2/4] Writing download manifest ...")
    write_manifest(download_root, raw_root / "download_manifest.txt")

    print("[3/4] Copying DAVIS / KIBA files into project structure ...")
    copied_davis, copied_kiba, unknown_files = copy_dataset_files(download_root, project_root)

    print(f"Copied DAVIS files: {copied_davis}")
    print(f"Copied KIBA files:  {copied_kiba}")

    if unknown_files:
        print(f"Found {len(unknown_files)} unclassified files, moving them to data/raw/_unclassified ...")
        n_misc = fallback_copy_unknowns(unknown_files, project_root)
        print(f"Copied unclassified files: {n_misc}")

    print("[4/4] Writing final manifests ...")
    write_manifest(project_root / "data" / "raw" / "davis", raw_root / "davis_manifest.txt")
    write_manifest(project_root / "data" / "raw" / "kiba", raw_root / "kiba_manifest.txt")

    print("\nDone.")
    print(f"DAVIS dir: {project_root / 'data' / 'raw' / 'davis'}")
    print(f"KIBA dir:  {project_root / 'data' / 'raw' / 'kiba'}")
    print("\n建议你接着执行：")
    print("  ls -R data/raw/davis")
    print("  ls -R data/raw/kiba")


if __name__ == "__main__":
    main()