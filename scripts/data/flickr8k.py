"""
The split token only works for imageid.human-written-caption.txt or imagedid.manually-translated-caption.txt
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
from loguru import logger
import typer
import json
import re
from . import dir_iter, zip_iter

app = typer.Typer()


@app.command()
def coco(files: List[str], imgpath: str = typer.Option(...)):
    imgpath: Path = Path(imgpath)
    if imgpath.is_dir():
        img_dict = set(dir_iter(imgpath))
    elif imgpath.suffix == ".zip":
        img_dict = set(zip_iter(imgpath))
    else:
        raise ValueError(f"Unsupported file type for imgpath: {imgpath}")

    files = [Path(f) for f in files]
    for file in files:
        if file.suffix != ".txt":
            logger.info(f"Skip {file}")
            continue
        logger.info(f"Processing {file}")
        samples = []
        with file.open('r', encoding="utf8") as rf:
            for line in rf:
                line = line.strip()
                try:
                    fields = line.split()
                    imgfile = fields[0]
                    imgfile = imgfile.split('#')[0]
                    caption = line[len(imgfile):].strip()
                except ValueError as e:
                    print(f"Error processing {line}")
                    raise e
                if imgfile not in img_dict:
                    logger.warning(f"Skip: {imgfile} not exists when parsing {line}")
                    continue
                samples.append({
                    "image": imgfile,
                    "image_id": Path(imgfile).stem,
                    "caption": caption,
                })
        # with (file.with_suffix(".json")).open('w', encoding="utf8") as wf:
        #     json.dump(samples, wf, ensure_ascii=False)


if __name__ == "__main__":
    app()
