"""Resumable public IXI archive download using eight bounded range requests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen
import shutil

URL = 'https://developer.download.nvidia.com/assets/Clara/monai/tutorials/IXI-T1.tar'
SIZE = 4840816640
CHUNK = 32 * 1024 * 1024
root = Path('data/ixi')
parts = root / 'download-parts'
parts.mkdir(parents=True, exist_ok=True)


def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    target = parts / f'{index:04d}.part'
    if target.exists() and target.stat().st_size == end - start + 1:
        return index
    request = Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/{SIZE}':
            raise ValueError('Server did not honor byte range')
        with target.open('wb') as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
    if target.stat().st_size != end - start + 1:
        raise ValueError('Incomplete range')
    return index


count = (SIZE + CHUNK - 1) // CHUNK
with ThreadPoolExecutor(max_workers=8) as pool:
    for completed, future in enumerate(as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
        future.result()
        if completed % 10 == 0 or completed == count:
            print(f'Downloaded {completed}/{count} chunks', flush=True)
combined = root / 'IXI-T1.verified-download'
with combined.open('wb') as destination:
    for index in range(count):
        with (parts / f'{index:04d}.part').open('rb') as source:
            shutil.copyfileobj(source, destination, 1024 * 1024)
with combined.open('rb') as stream:
    if hashlib.file_digest(stream, 'md5').hexdigest() != '34901a0593b41dd19c1a1f746eac2d58':
        raise ValueError('Archive checksum mismatch')
combined.replace(root / 'IXI-T1.tar')
print('Archive complete and checksum verified', flush=True)
