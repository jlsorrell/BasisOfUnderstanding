# scripts/fetch_glove.py
"""Download GloVe vectors into ./data once. Usage: python scripts/fetch_glove.py [100|300]"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://nlp.stanford.edu/data/glove.6B.zip"
DATA = Path("data")


def main(dim: str = "100") -> None:
    DATA.mkdir(exist_ok=True)
    target = DATA / f"glove.6B.{dim}d.txt"
    if target.exists():
        print(f"{target} already present.")
        return
    zip_path = DATA / "glove.6B.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} (~822 MB)...")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"Extracting glove.6B.{dim}d.txt...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(f"glove.6B.{dim}d.txt", DATA)
    print(f"Ready: {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "100")
