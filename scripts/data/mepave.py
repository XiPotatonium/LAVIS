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
def ner(files: List[str], imgpath: str = typer.Option(...)):
    imgpath: Path = Path(imgpath)
    if imgpath.is_dir():
        img_dict = set(dir_iter(imgpath))
    elif imgpath.suffix == ".zip":
        img_dict = set(zip_iter(imgpath))
    else:
        raise ValueError(f"Unsupported file type for imgpath: {imgpath}")

    tag_rule = re.compile(r"</?([^>]+)>")
    def labels_mapper(text: str, labels: str) -> List[Dict[str, Any]]:
        open_tags = []
        entities = []
        sent = ""
        last_end = 0
        for match in tag_rule.finditer(labels):
            sent += labels[last_end:match.start()]
            last_end = match.end()
            if match.group(0).startswith("</"):
                tag, start_idx = open_tags.pop()
                assert tag == match.group(1)            # intersection is not allowed
                entities.append({"start": start_idx, "end": len(sent), "type": tag})
            else:
                open_tags.append((match.group(1), len(sent)))
        sent += labels[last_end:]
        assert len(open_tags) == 0
        assert sent == text, f"{sent} vs {text}"
        return entities

    types = set()
    files = [Path(f) for f in files]
    for file in files:
        if file.suffix != ".txt":
            logger.info(f"Skip {file}")
            continue
        logger.info(f"Processing {file}")
        with file.open('r', encoding="utf8") as rf, \
            file.with_suffix(".jsonl").open('w', encoding="utf8") as wf:
            for line in rf:
                img, ident, text, label = line.strip().split('\t')
                img = img[2:] + ".jpg"
                assert img in img_dict
                entities = labels_mapper(text, label)
                types.update(entity["type"] for entity in entities)
                wf.write(json.dumps({
                    "img_name": img,
                    "id": ident,
                    "tokens": list(text),     # tokenize in chars
                    "entities": entities,
                    "raw_labels": label,
                }, ensure_ascii=False))
                wf.write('\n')
    logger.info(f"Get {len(types)} types")
    with files[0].with_name("meta.json").open('w', encoding="utf8") as wf:
        json.dump({
            "entities": {t: {"short": t, "verbose": t} for t in types}
        }, wf, ensure_ascii=False)


@app.command()
def coco(files: List[str], imgpath: str = typer.Option(...)):
    imgpath: Path = Path(imgpath)
    if imgpath.is_dir():
        img_dict = set(dir_iter(imgpath))
    elif imgpath.suffix == ".zip":
        img_dict = set(zip_iter(imgpath))
    else:
        raise ValueError(f"Unsupported file type for imgpath: {imgpath}")

    types = set()
    files = [Path(f) for f in files]
    for file in files:
        logger.info(f"Processing {file}")
        samples = []
        with file.open('r', encoding="utf8") as rf:
            for line in rf:
                img, ident, text, label = line.strip().split('\t')
                img = img[2:]
                img_file = img + ".jpg"
                assert img_file in img_dict
                samples.append({
                    "image": img_file,
                    "image_id": img,
                    "caption": text,
                })
        with (file.with_suffix(".json")).open('w', encoding="utf8") as wf:
            json.dump(samples, wf, ensure_ascii=False)
    logger.info(f"Get {len(types)} types")


if __name__ == "__main__":
    app()
