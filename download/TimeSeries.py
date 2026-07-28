import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
TIME_SERIES_ROOT = RAW_ROOT / "timeseries"

CHUNK_SIZE = 1024 * 1024
PROGRESS_EVERY_BYTES = 64 * 1024 * 1024
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _hf_tslib_url(path: str) -> str:
    return f"https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/{path}?download=true"


def _zenodo_pems_url(filename: str) -> str:
    return f"https://zenodo.org/records/7816008/files/{filename}?download=1"


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    target_relpath: Path
    urls: tuple[str, ...]
    kind: str
    min_bytes: int = 1
    md5: str | None = None

    @property
    def target_path(self) -> Path:
        return TIME_SERIES_ROOT / self.target_relpath


DOWNLOADS: tuple[DownloadSpec, ...] = (
    DownloadSpec(
        name="Electricity",
        target_relpath=Path("tslib") / "electricity" / "electricity.csv",
        urls=(_hf_tslib_url("electricity/electricity.csv"),),
        kind="tslib_csv",
        min_bytes=1024 * 1024,
    ),
    DownloadSpec(
        name="Weather",
        target_relpath=Path("tslib") / "weather" / "weather.csv",
        urls=(_hf_tslib_url("weather/weather.csv"),),
        kind="tslib_csv",
        min_bytes=1024 * 1024,
    ),
    DownloadSpec(
        name="Exchange",
        target_relpath=Path("tslib") / "exchange_rate" / "exchange_rate.csv",
        urls=(_hf_tslib_url("exchange_rate/exchange_rate.csv"),),
        kind="tslib_csv",
        min_bytes=1024,
    ),
    DownloadSpec(
        name="Traffic",
        target_relpath=Path("tslib") / "traffic" / "traffic.csv",
        urls=(_hf_tslib_url("traffic/traffic.csv"),),
        kind="tslib_csv",
        min_bytes=1024 * 1024,
    ),
    DownloadSpec(
        name="PEMS03 data",
        target_relpath=Path("pems") / "PEMS03" / "PEMS03.npz",
        urls=(_zenodo_pems_url("PEMS03.npz"),),
        kind="npz",
        min_bytes=1024 * 1024,
        md5="651add9bb9eaf7f5eda2f2ee8778a182",
    ),
    DownloadSpec(
        name="PEMS03 graph csv",
        target_relpath=Path("pems") / "PEMS03" / "PEMS03.csv",
        urls=(_zenodo_pems_url("PEMS03.csv"),),
        kind="csv",
        min_bytes=1024,
        md5="e6cf2ef1e1a9957aeec678a30b7ce7a7",
    ),
    DownloadSpec(
        name="PEMS03 distances",
        target_relpath=Path("pems") / "PEMS03" / "PEMS03.txt",
        urls=(_zenodo_pems_url("PEMS03.txt"),),
        kind="text",
        min_bytes=1024,
        md5="a52b864fbbe9d852ebdd08f6191a987f",
    ),
    DownloadSpec(
        name="PEMS04 data",
        target_relpath=Path("pems") / "PEMS04" / "PEMS04.npz",
        urls=(_zenodo_pems_url("PEMS04.npz"),),
        kind="npz",
        min_bytes=1024 * 1024,
        md5="5238832691101da85fcac772a32051da",
    ),
    DownloadSpec(
        name="PEMS04 graph csv",
        target_relpath=Path("pems") / "PEMS04" / "PEMS04.csv",
        urls=(_zenodo_pems_url("PEMS04.csv"),),
        kind="csv",
        min_bytes=1024,
        md5="3990731ec980019a875a6d5723523855",
    ),
    DownloadSpec(
        name="PEMS07 data",
        target_relpath=Path("pems") / "PEMS07" / "PEMS07.npz",
        urls=(_zenodo_pems_url("PEMS07.npz"),),
        kind="npz",
        min_bytes=1024 * 1024,
        md5="978d3d9b85fe640a446983a34271a48d",
    ),
    DownloadSpec(
        name="PEMS07 graph csv",
        target_relpath=Path("pems") / "PEMS07" / "PEMS07.csv",
        urls=(_zenodo_pems_url("PEMS07.csv"),),
        kind="csv",
        min_bytes=1024,
        md5="cea609fed853ade176cb1c5eae0aeda0",
    ),
    DownloadSpec(
        name="PEMS08 data",
        target_relpath=Path("pems") / "PEMS08" / "PEMS08.npz",
        urls=(_zenodo_pems_url("PEMS08.npz"),),
        kind="npz",
        min_bytes=1024 * 1024,
        md5="2a528d169c0d90294c9e24288a430132",
    ),
    DownloadSpec(
        name="PEMS08 graph csv",
        target_relpath=Path("pems") / "PEMS08" / "PEMS08.csv",
        urls=(_zenodo_pems_url("PEMS08.csv"),),
        kind="csv",
        min_bytes=1024,
        md5="5e507b2b18d9235e95b63a20221cd4c5",
    ),
)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_csv(
    path: Path,
    require_date_column: bool = False,
    require_data_row: bool = True,
) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty or missing a header row.")
        if require_date_column and "date" not in header:
            raise ValueError("TSLib CSV is expected to contain a 'date' column.")
        first_data_row = next(reader, None)
        if require_data_row and not first_data_row:
            raise ValueError("CSV has no data rows.")


def _validate_npz(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if not magic.startswith(b"PK"):
        raise ValueError("NPZ file does not look like a zip container.")


def _validate_archive(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(512)
    archive_magics = (
        b"PK",
        b"\x1f\x8b",
        b"BZh",
        b"7z\xbc\xaf\x27\x1c",
    )
    if header.startswith(archive_magics):
        return
    if len(header) > 265 and header[257:262] == b"ustar":
        return
    raise ValueError("Downloaded file does not look like a supported archive.")


def _validate_download(spec: DownloadSpec, path: Path | None = None) -> None:
    path = spec.target_path if path is None else path
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < spec.min_bytes:
        raise ValueError(f"File is too small: {size} bytes < {spec.min_bytes} bytes.")

    if spec.md5 is not None:
        actual = _file_md5(path)
        if actual != spec.md5:
            raise ValueError(f"MD5 mismatch: expected {spec.md5}, got {actual}.")

    if spec.kind == "tslib_csv":
        _validate_csv(path, require_date_column=True)
    elif spec.kind == "csv":
        _validate_csv(path, require_date_column=False, require_data_row=False)
    elif spec.kind == "text":
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError("Text file is empty.")
    elif spec.kind == "npz":
        _validate_npz(path)
    elif spec.kind == "archive":
        _validate_archive(path)
    else:
        raise ValueError(f"Unknown download kind: {spec.kind}")


def _download_to_file(
    url: str,
    out_path: Path,
    timeout: int,
    retries: int,
    retry_backoff: float,
) -> None:
    last_exc = None
    total_attempts = max(1, retries)

    for attempt in range(1, total_attempts + 1):
        bytes_written = 0
        next_progress = PROGRESS_EVERY_BYTES
        try:
            request = Request(url, headers={"User-Agent": "sdta-timeseries-downloader/1.0"})
            with urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise RuntimeError(f"HTTP status {status} for {url}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bytes_written += len(chunk)
                        if bytes_written >= next_progress:
                            mib = bytes_written / (1024 * 1024)
                            print(f"  downloaded {mib:.0f} MiB")
                            next_progress += PROGRESS_EVERY_BYTES
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


def download_dataset(
    spec: DownloadSpec,
    force: bool = False,
    timeout: int = 60,
    retries: int = 3,
    retry_backoff: float = 1.5,
) -> Path:
    target_path = spec.target_path
    if target_path.exists() and not force:
        try:
            _validate_download(spec)
            print(f"Skip valid file: {target_path}")
            return target_path
        except (OSError, ValueError) as exc:
            print(f"Existing file is invalid, redownloading: {target_path} ({exc})")

    tmp_path = target_path.with_name(target_path.name + ".part")
    errors = []

    for url in spec.urls:
        try:
            if tmp_path.exists():
                tmp_path.unlink()

            print(f"Downloading {spec.name}: {url}")
            _download_to_file(
                url=url,
                out_path=tmp_path,
                timeout=timeout,
                retries=retries,
                retry_backoff=retry_backoff,
            )
            _validate_download(spec, tmp_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(target_path)
            print(f"Saved: {target_path}")
            return target_path
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{url} -> {exc}")

    if tmp_path.exists():
        tmp_path.unlink()

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to download {spec.name} from all sources:\n{joined}")


def download_all(downloads: Iterable[DownloadSpec] = DOWNLOADS) -> list[Path]:
    saved_paths = []
    for spec in downloads:
        saved_paths.append(download_dataset(spec))
    return saved_paths


def main() -> None:
    saved_paths = download_all()
    print(f"Done. {len(saved_paths)} files are available under {TIME_SERIES_ROOT}")


if __name__ == "__main__":
    main()
