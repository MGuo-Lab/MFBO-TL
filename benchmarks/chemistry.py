"""Retrieve the three chemistry pools from their versioned source archive."""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://zenodo.org/records/15269206/files/chem-MFBO-main.zip?download=1'
MD5 = '25aef4153efcb20653002f620f806990'


def prepare(outdir):
    archive = urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.md5(archive).hexdigest() != MD5:
        raise ValueError('Source archive checksum does not match the versioned record.')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for name in ['cofs', 'freesolv', 'polarizability']:
            path = outdir / f'{name}.csv'
            path.write_bytes(source.read(f'chem-MFBO-main/data/clean/{name}.csv'))
            print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    prepare(parser.parse_args().outdir)
