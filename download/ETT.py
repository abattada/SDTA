import argparse
import csv
import time
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = "ETT-small"
SUPPORTED_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}
DEFAULT_URL_TEMPLATES = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/{dataset_dir}/{dataset}.csv",
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/master/{dataset_dir}/{dataset}.csv",
    "https://cdn.jsdelivr.net/gh/zhouhaoyi/ETDataset@main/{dataset_dir}/{dataset}.csv",
)

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _resolve_path(path_value: object, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return base_dir / path


def _validate_csv(csv_path: Path) -> int:
    """Basic validity check for ETT schema and row count."""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Downloaded file is empty.") from exc

        if "OT" not in header:
            raise ValueError("Downloaded file does not contain required 'OT' column.")

        row_count = 0
        for _ in reader:
            row_count += 1

    if row_count == 0:
        raise ValueError("Downloaded file has no data rows.")
    return row_count


def _download_to_file(url: str, out_path: Path, timeout: int, retries: int, retry_backoff: float) -> None:
    last_exc = None
    total_attempts = max(1, retries)

    for attempt in range(1, total_attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "sdta-ett-downloader/1.0"})
            with urlopen(req, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise RuntimeError(f"HTTP status {status} for {url}")

                with out_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 64)
                        if not chunk:
                            break
                        f.write(chunk)
                return
        except HTTPError as exc:
            last_exc = exc
            if exc.code in RETRYABLE_HTTP_CODES and attempt < total_attempts:
                print(f"Retrying ({attempt}/{total_attempts}) after HTTP {exc.code}: {url}")
                time.sleep(retry_backoff * attempt)
                continue
            raise
        except (URLError, TimeoutError, RuntimeError) as exc:
            last_exc = exc
            if attempt < total_attempts:
                print(f"Retrying ({attempt}/{total_attempts}) after error: {url} ({exc})")
                time.sleep(retry_backoff * attempt)
                continue
            raise

    if last_exc is not None:
        raise last_exc


def _build_dataset_urls(
    dataset: str,
    dataset_dir: str,
    extra_urls: Iterable[str] = (),
    url_templates: Iterable[str] = DEFAULT_URL_TEMPLATES,
) -> List[str]:
    urls = [url.format(dataset=dataset, dataset_dir=dataset_dir) for url in url_templates]
    return list(extra_urls) + urls


def download_ett_dataset(
    dataset: str,
    out_dir: Path,
    force: bool = False,
    timeout: int = 30,
    retries: int = 3,
    retry_backoff: float = 1.5,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    urls: Iterable[str] = (),
) -> Path:
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"不支援的 ETT dataset: {dataset}. 可用: {supported}")

    out_dir.mkdir(parents=True, exist_ok=True)
    target_csv = out_dir / f"{dataset}.csv"

    if target_csv.exists() and not force:
        try:
            rows = _validate_csv(target_csv)
            print(f"File already exists and is valid: {target_csv} ({rows} rows)")
            return target_csv
        except ValueError:
            print(f"Existing file is invalid, redownloading: {target_csv}")

    tmp_csv = out_dir / f"{dataset}.csv.part"
    errors = []

    for url in urls:
        try:
            if tmp_csv.exists():
                tmp_csv.unlink()

            print(f"Downloading from: {url}")
            _download_to_file(
                url=url,
                out_path=tmp_csv,
                timeout=timeout,
                retries=retries,
                retry_backoff=retry_backoff,
            )
            rows = _validate_csv(tmp_csv)
            tmp_csv.replace(target_csv)
            print(f"Saved: {target_csv} ({rows} rows)")
            return target_csv
        except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
            errors.append(f"{url} -> {exc}")

    if tmp_csv.exists():
        tmp_csv.unlink()

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to download {dataset}.csv from all sources:\n{joined}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download an ETT dataset into data/raw/")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists")
    parser.add_argument("--timeout", type=int, default=30, help="Download timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per URL")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Retry backoff multiplier")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Additional URL(s) to try first; can be used multiple times",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    merged_urls = _build_dataset_urls(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        extra_urls=tuple(args.url),
    )

    download_ett_dataset(
        dataset=args.dataset,
        out_dir=_resolve_path(args.out_dir),
        force=args.force,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        dataset_dir=args.dataset_dir,
        urls=merged_urls,
    )


if __name__ == "__main__":
    main()
