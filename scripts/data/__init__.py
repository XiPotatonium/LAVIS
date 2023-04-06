from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
from loguru import logger


def dir_iter(folder: Path) -> Iterator[str]:
    valid_extensions = {".jpg"}
    for file in folder.iterdir():
        if file.is_file() and file.suffix in valid_extensions:
            yield file.name
        else:
            logger.warning(f"Not supported file type {file}")


def zip_iter(file: Path) -> Iterator[str]:
    import zipfile
    with zipfile.ZipFile(file, 'r') as archive:
        for fpath in archive.namelist():
            if fpath.endswith('.jpg'):
                yield Path(fpath).name