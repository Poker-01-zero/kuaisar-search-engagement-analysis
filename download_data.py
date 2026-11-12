"""Download the official KuaiSAR-small archive and verify its published MD5."""
from pathlib import Path
import hashlib
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
TARGET = RAW / 'KuaiSAR.zip'
URL = 'https://zenodo.org/records/8181109/files/KuaiSAR.zip?download=1'
EXPECTED = 'daea8cbf605db6bd5841740f0e4a12d9'

if not TARGET.exists():
    part = TARGET.with_suffix('.zip.part')
    started = time.monotonic()
    size = part.stat().st_size if part.exists() else 0
    headers = {'Range': f'bytes={size}-'} if size else {}
    with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if response.status_code == 206:
            if not response.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
                raise RuntimeError('Invalid resume Content-Range')
            mode = 'ab'
        else:
            size = 0
            mode = 'wb'
        last = time.monotonic()
        with part.open(mode) as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
                size += len(chunk)
                if time.monotonic() - last >= 15:
                    print(f'Downloaded {size / 1e6:.1f} MB', flush=True)
                    last = time.monotonic()
    with part.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'md5').hexdigest()
    if digest != EXPECTED:
        raise RuntimeError(f'Checksum mismatch before rename: {digest}')
    part.rename(TARGET)
    print(f'Download seconds: {time.monotonic() - started:.1f}', flush=True)
with TARGET.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'md5').hexdigest()
if digest != EXPECTED:
    raise RuntimeError(f'Checksum mismatch: {digest}')
print(f'MD5 verified: {digest}', flush=True)
with zipfile.ZipFile(TARGET) as archive:
    for info in archive.infolist():
        destination = (RAW / info.filename).resolve()
        if not destination.is_relative_to(RAW.resolve()):
            raise ValueError('Unsafe archive path')
        print(f'{info.filename}: {info.file_size:,} bytes', flush=True)
        if not destination.exists() or destination.stat().st_size != info.file_size:
            archive.extract(info, RAW)
print('Extraction complete', flush=True)
