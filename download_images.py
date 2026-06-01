"""
Download all EMB images listed in early_melanoma_benchmark_dataset_labels.csv.

Data sources (per https://doi.org/10.3390/cancers17152476):
  - ISIC Archive (907 images): public S3 bucket
  - Dermoscopy Atlas (197 images): https://www.dermoscopyatlas.com/uploads/cases/
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

ISIC_S3_URL = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"
ATLAS_URL = "https://www.dermoscopyatlas.com/uploads/cases/{image_id}.jpg"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class DownloadResult:
    image_id: str
    source: str
    status: str
    path: str | None = None
    error: str | None = None


def download_image(url: str, output_path: Path, timeout: int = 120) -> None:
    if output_path.exists() and output_path.stat().st_size > 1024:
        return

    response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                handle.write(chunk)
    temp_path.replace(output_path)


def fetch_one(image_id: str, source: str, output_dir: Path) -> DownloadResult:
    output_path = output_dir / f"{image_id}.jpg"
    if output_path.exists() and output_path.stat().st_size > 1024:
        return DownloadResult(image_id, source, "skipped", str(output_path))

    url = ISIC_S3_URL.format(image_id=image_id) if source == "ISIC" else ATLAS_URL.format(
        image_id=image_id
    )
    try:
        download_image(url, output_path)
        return DownloadResult(image_id, source, "ok", str(output_path))
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return DownloadResult(image_id, source, "failed", error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all EMB benchmark images.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "early_melanoma_benchmark_dataset_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--source",
        type=str,
        default="ISIC",
        help="Download only this source (default: ISIC, 907 images).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    df = df[df["source"].str.upper() == args.source.upper()]
    records = df[["image", "source"]].drop_duplicates("image")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(records)} images to {args.output_dir} ...")
    started = time.time()
    results: list[DownloadResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, row.image, row.source, args.output_dir): row.image
            for row in records.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 50 == 0 or index == len(futures):
                ok = sum(r.status == "ok" for r in results)
                skipped = sum(r.status == "skipped" for r in results)
                failed = sum(r.status == "failed" for r in results)
                elapsed = time.time() - started
                print(
                    f"[{index}/{len(futures)}] ok={ok} skipped={skipped} "
                    f"failed={failed} elapsed={elapsed:.0f}s"
                )

    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]

    manifest = {
        "labels_csv": str(args.labels_csv),
        "output_dir": str(args.output_dir),
        "total": len(results),
        "downloaded": len(ok),
        "skipped": len(skipped),
        "failed": len(failed),
        "failures": [{"image": r.image_id, "source": r.source, "error": r.error} for r in failed],
    }
    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nDone in {time.time() - started:.0f}s: "
        f"{len(ok)} downloaded, {len(skipped)} skipped, {len(failed)} failed."
    )
    print(f"Manifest: {manifest_path}")
    if failed:
        print(f"First failures: {failed[:5]}")


if __name__ == "__main__":
    main()
