"""Download publicly licensed IXI MRI data; verify archive integrity before use."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

root = Path('data/ixi')
root.mkdir(parents=True, exist_ok=True)
sources = {
    'IXI-T1.tar': 'https://developer.download.nvidia.com/assets/Clara/monai/tutorials/IXI-T1.tar',
    'IXI.xls': 'https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls',
}
for filename, url in sources.items():
    destination = root / filename
    if not destination.exists():
        partial = root / (filename + '.part')
        with urlopen(url, timeout=60) as response, partial.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        partial.rename(destination)
with (root / 'IXI-T1.tar').open('rb') as stream:
    if hashlib.file_digest(stream, 'md5').hexdigest() != '34901a0593b41dd19c1a1f746eac2d58':
        raise ValueError('Archive checksum mismatch; do not use these data')
print('IXI archive verified. Source: https://brain-development.org/ixi-dataset/; CC BY-SA 3.0')
