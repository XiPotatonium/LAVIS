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
def check_overlap(files: List[str]):
    all_set = set()
    for file in files:
        with Path(file).open('r', encoding="utf8") as rf:
            samples = json.load(rf)
        fset = set([s["image"] for s in samples])
        print(f"{file}: {len(samples)} samples, {len(fset)} images")
        all_set.update(fset)
    print(f"all set size: {len(all_set)}")


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
                    img = fields[0]
                    img = img.split('#')[0]
                    caption = line[len(img):].strip()
                except ValueError as e:
                    print(f"Error processing {line}")
                    raise e
                imgfile = img + ".jpg"
                assert imgfile in img_dict
                samples.append({
                    "image": imgfile,
                    "image_id": img,
                    "caption": caption,
                })
        with (file.with_suffix(".json")).open('w', encoding="utf8") as wf:
            json.dump(samples, wf, ensure_ascii=False)


@app.command()
def stat(paths: List[str]):
    for path in paths:
        with Path(path).open('r', encoding="utf8") as rf:
            samples = json.load(rf)
        print("Got {} samples in {}".format(len(samples), path))
        max_len = max([len(s["caption"]) for s in samples])
        print(max_len)


if __name__ == "__main__":
    app()