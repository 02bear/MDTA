#!/usr/bin/env python3
"""
从 PubChem 自动下载 Davis 数据集药物的 SDF 文件。

默认读取:
  data/raw/davis/drugs.csv
默认输出:
  data/raw/davis/pubchem_sdf

drugs.csv 需要至少包含 drug_id 和 smiles 两列。
脚本会用 SMILES 在 PubChem 查询对应的 CID，再按 CID 下载 SDF。
输出 SDF 文件名使用 Davis 原始 drug_id，方便后续和数据集对齐。
优先下载 3D SDF；如果 PubChem 没有 3D 构象，则自动回退到 2D SDF。
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PUBCHEM_CID_BY_SMILES_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT"
PUBCHEM_SDF_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type={record_type}"
USER_AGENT = "MyModel-MDTA/1.0 (Davis SDF downloader; PubChem PUG-REST)"


@dataclass(frozen=True)
class DrugRecord:
    drug_id: str
    smiles: str


def project_root_from_script() -> Path:
    """返回 MyModel-MDTA 项目根目录。"""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_drug_records(drugs_csv: Path) -> list[DrugRecord]:
    if not drugs_csv.exists():
        raise FileNotFoundError(f"找不到 drugs.csv: {drugs_csv}")

    records: list[DrugRecord] = []
    seen: set[str] = set()

    with drugs_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"drug_id", "smiles"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{drugs_csv} 中需要包含 {sorted(required_columns)} 两列，实际列名: {reader.fieldnames}"
            )

        for row in reader:
            drug_id = str(row.get("drug_id", "")).strip()
            smiles = str(row.get("smiles", "")).strip()
            if not drug_id or not smiles or drug_id in seen:
                continue
            seen.add(drug_id)
            records.append(DrugRecord(drug_id=drug_id, smiles=smiles))

    if not records:
        raise ValueError(f"{drugs_csv} 中没有读到任何有效的 drug_id/smiles 记录")

    return records


def fetch_url(url: str, timeout: int, retries: int, sleep_seconds: float) -> bytes:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    assert last_error is not None
    raise last_error


def query_cid_by_smiles(smiles: str, timeout: int, retries: int, sleep_seconds: float) -> str:
    """用 SMILES 查询 PubChem CID，返回第一个匹配 CID。"""
    url = PUBCHEM_CID_BY_SMILES_URL.format(smiles=quote(smiles, safe=""))
    data = fetch_url(url, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
    cids = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
    if not cids:
        raise ValueError("PubChem 没有返回 CID")
    return cids[0]


def download_sdf_by_cid(
    cid: str,
    out_file: Path,
    timeout: int,
    retries: int,
    sleep_seconds: float,
    prefer_3d: bool,
) -> tuple[bool, str]:
    """
    按 CID 下载 SDF 到指定文件。

    返回:
      (是否成功, 下载结果说明)
    """
    if out_file.exists() and out_file.stat().st_size > 0:
        return True, "exists"

    record_types: Iterable[str]
    if prefer_3d:
        record_types = ("3d", "2d")
    else:
        record_types = ("2d",)

    errors: list[str] = []
    for record_type in record_types:
        url = PUBCHEM_SDF_URL.format(cid=cid, record_type=record_type)
        try:
            data = fetch_url(url, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
            if not data.strip():
                errors.append(f"{record_type}: empty response")
                continue
            out_file.write_bytes(data)
            return True, record_type
        except HTTPError as exc:
            errors.append(f"{record_type}: HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001 - CLI 脚本需要记录失败原因并继续后续药物
            errors.append(f"{record_type}: {type(exc).__name__}: {exc}")

    return False, "; ".join(errors)


def download_one_drug(
    record: DrugRecord,
    out_dir: Path,
    timeout: int,
    retries: int,
    sleep_seconds: float,
    prefer_3d: bool,
) -> tuple[bool, str, str]:
    """
    用 SMILES 查询 CID 并下载 SDF。

    返回:
      (是否成功, PubChem CID, 结果说明)
    """
    try:
        cid = query_cid_by_smiles(record.smiles, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
    except Exception as exc:  # noqa: BLE001 - CLI 脚本需要记录失败原因并继续后续药物
        return False, "", f"query_cid_failed: {type(exc).__name__}: {exc}"

    out_file = out_dir / f"{record.drug_id}.sdf"
    ok, detail = download_sdf_by_cid(
        cid=cid,
        out_file=out_file,
        timeout=timeout,
        retries=retries,
        sleep_seconds=sleep_seconds,
        prefer_3d=prefer_3d,
    )
    return ok, cid, detail


def write_report(report_file: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    with report_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_id", "smiles", "pubchem_cid", "status", "detail"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Download Davis drug SDF files from PubChem by SMILES.")
    parser.add_argument(
        "--drugs-csv",
        type=Path,
        default=root / "data" / "raw" / "davis" / "drugs.csv",
        help="Davis drugs.csv 路径，需包含 drug_id 和 smiles 两列。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "raw" / "davis" / "pubchem_sdf",
        help="SDF 输出目录。",
    )
    parser.add_argument("--timeout", type=int, default=30, help="单次请求超时时间，单位秒。")
    parser.add_argument("--retries", type=int, default=3, help="每个请求的重试次数。")
    parser.add_argument("--sleep", type=float, default=0.25, help="每个药物下载后的间隔，避免请求过快。")
    parser.add_argument(
        "--only-2d",
        action="store_true",
        help="只下载 2D SDF；默认优先下载 3D，失败后回退到 2D。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    records = read_drug_records(args.drugs_csv)
    print(f"从 {args.drugs_csv} 读取到 {len(records)} 个唯一 drug_id/smiles 记录")
    print(f"SDF 输出目录: {args.out_dir}")

    rows: list[tuple[str, str, str, str, str]] = []
    success = 0
    failed = 0

    for idx, record in enumerate(records, start=1):
        ok, cid, detail = download_one_drug(
            record=record,
            out_dir=args.out_dir,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
            prefer_3d=not args.only_2d,
        )
        if ok:
            success += 1
            status = "ok"
        else:
            failed += 1
            status = "failed"

        rows.append((record.drug_id, record.smiles, cid, status, detail))
        cid_text = cid if cid else "NA"
        print(
            f"[{idx:03d}/{len(records):03d}] drug_id {record.drug_id} -> PubChem CID {cid_text}: "
            f"{status} ({detail})"
        )
        time.sleep(args.sleep)

    report_file = args.out_dir / "download_report.csv"
    write_report(report_file, rows)

    print("完成")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"报告: {report_file}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
